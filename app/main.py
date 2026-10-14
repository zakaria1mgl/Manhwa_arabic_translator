"""Entry point: python -m app.main [--workdir DIR] [--config config.yaml] [--port 7860]"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Manhwa EN->AR Translator")
    parser.add_argument("--workdir", default=".", help="working directory")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Gradio public link")
    parser.add_argument("--chapter", default="default", help="initial chapter id")
    args = parser.parse_args()

    from .config import AppConfig, setup_logging
    setup_logging()
    log = logging.getLogger(__name__)

    os.makedirs(args.workdir, exist_ok=True)
    cfg = AppConfig.load(args.config)

    # pre-download fonts so first translation is fast
    try:
        from .typography.fonts import ensure_fonts
        fonts = cfg.get("typography", "fonts", default={}) or {}
        ensure_fonts(os.path.join(args.workdir,
                                  cfg.get("typography", "font_download_dir", default="fonts")),
                     list(fonts.values()) or None)
    except Exception as e:  # pragma: no cover
        log.warning("font pre-download failed: %s", e)

    from .ui.gradio_app import GradioApp
    app = GradioApp(cfg, workdir=args.workdir)
    app.chapter_id = args.chapter
    demo = app.build()
    log.info("launching UI on port %d", args.port)
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0" if os.environ.get("MANHWA_PUBLIC") else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
