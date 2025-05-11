#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import time
from datetime import datetime

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

# Silence detection settings
SILENCE_THRESHOLD = 10  # seconds before triggering a prompt
MAX_SILENT_PROMPTS = 3  # max number of silence prompts before ending call


class CallStatsTracker:
    """Tracks call statistics."""

    def __init__(self):
        self.start_time = time.time()
        self.silence_events = []
        self.unanswered_prompts = 0
        self.last_user_activity = time.time()
        self.total_user_messages = 0
        self.total_bot_messages = 0
        
    def update_user_activity(self):
        """Mark that the user was active (reset silence timer)."""
        self.last_user_activity = time.time()
        self.total_user_messages += 1
        
    def record_silence_prompt(self):
        """Record that a silence prompt was sent."""
        silence_duration = time.time() - self.last_user_activity
        self.silence_events.append({
            "timestamp": datetime.now().isoformat(),
            "duration": round(silence_duration, 1)
        })
        self.unanswered_prompts += 1
        
    def record_bot_message(self):
        """Record that the bot sent a message."""
        self.total_bot_messages += 1
        # Reset unanswered prompts if user responds
        if self.total_user_messages > 0:
            self.unanswered_prompts = 0
        
    def get_call_duration(self):
        """Get the current call duration in seconds."""
        return round(time.time() - self.start_time, 1)
        
    def generate_summary(self):
        """Generate a summary of the call statistics."""
        summary = {
            "duration_seconds": self.get_call_duration(),
            "duration_formatted": self.format_duration(self.get_call_duration()),
            "silence_events": len(self.silence_events),
            "silence_details": self.silence_events,
            "total_user_messages": self.total_user_messages,
            "total_bot_messages": self.total_bot_messages,
        }
        return summary
    
    @staticmethod
    def format_duration(seconds):
        """Format duration in seconds to minutes:seconds."""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}:{seconds:02d}"


async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager and stats tracker
    session_manager = SessionManager()
    stats_tracker = CallStatsTracker()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Silence Detection Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Log call statistics before ending
        logger.info(f"Call ended with stats: {stats_tracker.generate_summary()}")
        
        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ SILENCE DETECTION SETUP ------------
    
    async def check_silence():
        """Periodically check for silence and send prompts if needed."""
        while True:
            await asyncio.sleep(1)  # Check every second
            
            # Skip if call is already terminated
            if session_manager.call_flow_state.call_terminated:
                break
                
            # Calculate time since last user activity
            silence_duration = time.time() - stats_tracker.last_user_activity
            
            # If we've reached the silence threshold and haven't exceeded max prompts
            if silence_duration >= SILENCE_THRESHOLD and stats_tracker.unanswered_prompts < MAX_SILENT_PROMPTS:
                logger.info(f"Silence detected for {silence_duration:.1f} seconds, sending prompt")
                stats_tracker.record_silence_prompt()
                
                # Different messages based on how many silent prompts we've sent
                if stats_tracker.unanswered_prompts == 1:
                    prompt = "I noticed you've been quiet. Is there anything you'd like to discuss?"
                elif stats_tracker.unanswered_prompts == 2:
                    prompt = "Are you still there? I'm ready to help whenever you're ready."
                else:
                    prompt = "Since I haven't heard from you, I'll end the call shortly if there's no response."
                
                # Create a text frame for the silence prompt
                silence_frame = TextFrame(text=prompt)
                
                # Send the frame to TTS and then to transport output
                await tts.process_frame(silence_frame, FrameDirection.DOWNSTREAM)
                
                # Reset the timer to avoid multiple prompts
                stats_tracker.last_user_activity = time.time()
            
            # If we've reached max unanswered prompts, terminate the call
            elif stats_tracker.unanswered_prompts >= MAX_SILENT_PROMPTS:
                logger.info(f"Maximum silent prompts ({MAX_SILENT_PROMPTS}) reached, terminating call")
                
                # Say goodbye before ending
                goodbye = "Since I haven't heard from you, I'll end our call now. Goodbye!"
                goodbye_frame = TextFrame(text=goodbye)
                await tts.process_frame(goodbye_frame, FrameDirection.DOWNSTREAM)
                
                # Wait for the goodbye to be spoken
                await asyncio.sleep(5)
                
                # Then terminate
                await terminate_call(FunctionCallParams(llm=llm))
                break

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        
        # Start the silence detection coroutine
        asyncio.create_task(check_silence())

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.debug(f"Participant left: {participant}, reason: {reason}")
        
        # Log call statistics before ending
        logger.info(f"Call ended with stats: {stats_tracker.generate_summary()}")
        
        await task.cancel()

    @transport.event_handler("on_transcription")
    async def on_transcription(transport, participant_id, transcription):
        """Handle transcription events to track user activity."""
        if transcription and transcription.text:
            # Update the last activity time when user speaks
            stats_tracker.update_user_activity()
            logger.debug(f"User activity detected: {transcription.text}")

    @context_aggregator.assistant().event_handler("on_process_frame")
    async def on_assistant_message(processor, frame, direction):
        """Track bot messages for statistics."""
        if frame and hasattr(frame, "text") and frame.text:
            stats_tracker.record_bot_message()
            logger.debug(f"Bot message: {frame.text}")

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silence Detection Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body)) 