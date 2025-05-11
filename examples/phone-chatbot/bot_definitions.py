# bot_definitions.py
"""Definitions of different bot types for the bot registry."""

from bot_registry import BotRegistry, BotRegistration
from bot_runner_helpers import (
    create_call_transfer_settings,
    create_simple_dialin_settings,
    create_simple_dialout_settings,
)

# Create the registry
bot_registry = BotRegistry()

# Register the bots
bot_registry.register(
    BotRegistration(
        name="simple_dialin",
        description="Simple dial-in bot with no additional features",
        module="simple_dialin",
        config_keys=["simple_dialin"],
    )
)

bot_registry.register(
    BotRegistration(
        name="simple_dialout",
        description="Simple dial-out bot with no additional features",
        module="simple_dialout",
        config_keys=["simple_dialout"],
    )
)

bot_registry.register(
    BotRegistration(
        name="voicemail_detection",
        description="Bot that can detect whether it reached voicemail or a person and respond accordingly",
        module="voicemail_detection",
        config_keys=["voicemail_detection"],
    )
)

bot_registry.register(
    BotRegistration(
        name="call_transfer",
        description="Bot that can transfer calls to human operators",
        module="call_transfer",
        config_keys=["call_transfer"],
    )
)

# Register our new silence detection bot
bot_registry.register(
    BotRegistration(
        name="silence_detection_bot",
        description="Bot with silence detection, graceful termination, and call statistics",
        module="silence_detection_bot",
        config_keys=["silence_detection"],
    )
)
