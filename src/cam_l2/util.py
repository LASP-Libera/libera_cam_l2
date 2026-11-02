#╭────────────────────────────────────────────────────────────────────────────╮!
# @LICENSE_HEADER_START@
#
#   This file is part of L2_CAM.
#
#   --
#   L2_CAM: Code for Processing Libera Camera L2 Products
#   --
#
#   Copyright (C) 2025-
#   EaR³T Community: https://discord.gg/ntqsguwaWv
#
#   L2_CAM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   L2_CAM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with L2_CAM. If not, see <http://www.gnu.org/licenses/>.
#
# @LICENSE_HEADER_END@
#
# History:
#   Created by: Hong Chen
#               hong.chen@hchen.com
#╰────────────────────────────────────────────────────────────────────────────╯!

import yaml
import colorama
import structlog

__all__ = ['read_cfg', 'get_logger']


def read_cfg(fname):

    with open(fname, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_logger():

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="%H:%M:%S", key="timestamp", utc=False),

            # Log level (colored)
            structlog.stdlib.add_log_level,

            # Module + line number
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),

            structlog.stdlib.PositionalArgumentsFormatter(),

            structlog.processors.format_exc_info,

            structlog.dev.ConsoleRenderer(
                columns=[

                    # Render the timestamp without the key name in yellow.
                    structlog.dev.Column(
                        "timestamp",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=None,
                            value_style=colorama.Style.RESET_ALL,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                        ),
                    ),

                    # Render the event without the key name in bright magenta.
                    structlog.dev.Column(
                        "level",
                        structlog.dev.LogLevelColumnFormatter(
                            level_styles={
                                'info' : colorama.Fore.GREEN,
                                'warn' : colorama.Fore.WHITE+colorama.Back.YELLOW,
                                'error': colorama.Fore.WHITE+colorama.Back.RED,
                                'debug': colorama.Fore.CYAN,
                                },
                            reset_style=colorama.Style.RESET_ALL,
                            ),
                    ),

                    # Render the event without the key name in bright magenta.
                    structlog.dev.Column(
                        "event",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=None,
                            value_style=colorama.Fore.LIGHTWHITE_EX,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                            postfix='\n'
                        ),
                    ),

                    structlog.dev.Column(
                        "filename",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=colorama.Fore.BLACK,
                            value_style=colorama.Fore.BLACK,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                            prefix='                ',
                            width=12
                        ),
                    ),

                    structlog.dev.Column(
                        "lineno",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=colorama.Fore.BLACK,
                            value_style=colorama.Fore.BLACK,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                            width=5
                        ),
                    ),

                    structlog.dev.Column(
                        "func_name",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=colorama.Fore.BLACK,
                            value_style=colorama.Fore.BLACK,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                            postfix=colorama.Style.RESET_ALL,
                        ),
                    ),

                    structlog.dev.Column(
                        "",
                        structlog.dev.KeyValueColumnFormatter(
                            key_style=colorama.Style.RESET_ALL,
                            value_style=colorama.Style.RESET_ALL,
                            reset_style=colorama.Style.RESET_ALL,
                            value_repr=str,
                        ),
                    ),
                ]
            )

        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        )

    logger = structlog.get_logger()

    return logger

if __name__ == "__main__":

    pass
