"""Command-line interface for ImageRouter."""

from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn

from .client import ImageRouterClient
from .estimator import CostEstimator
from .exceptions import ImageRouterError
from .generators import ImageGenerator, VideoGenerator
from .models import ModelRegistry


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="imagerouter",
        description="ImageRouter Video/Image Generation CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Estimate command
    estimate_parser = subparsers.add_parser(
        "estimate",
        help="Estimate generation cost (default mode, no API cost)",
    )
    _add_estimate_args(estimate_parser)

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Execute generation (requires --execute flag)",
    )
    _add_generate_args(generate_parser)

    # Models command
    models_parser = subparsers.add_parser(
        "models",
        help="List available models",
    )
    _add_models_args(models_parser)

    # Credits command
    credits_parser = subparsers.add_parser(
        "credits",
        help="Show account balance",
    )
    _add_credits_args(credits_parser)

    return parser


def _add_estimate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the estimate command."""
    parser.add_argument(
        "--type",
        choices=["video", "image"],
        required=True,
        help="Generation type",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID (e.g., 'google/veo-3.1-fast')",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        help="Video duration in seconds",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of outputs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="auto",
        help="Image quality level (default: auto)",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output resolution (default: auto)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the generate command."""
    parser.add_argument(
        "--execute",
        action="store_true",
        required=True,
        help="Required flag to confirm generation",
    )
    parser.add_argument(
        "--type",
        choices=["video", "image"],
        required=True,
        help="Generation type",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--image",
        action="append",
        help="Input image path(s) for i2v/i2i (can be specified multiple times)",
    )
    parser.add_argument(
        "--mask",
        action="append",
        help="Mask image path(s) for editing (can be specified multiple times)",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        help="Video duration in seconds",
    )
    parser.add_argument(
        "--size",
        default="auto",
        help="Output resolution",
    )
    parser.add_argument(
        "--quality",
        default="auto",
        help="Image quality level",
    )
    parser.add_argument(
        "--output",
        help="Save output to file path",
    )
    parser.add_argument(
        "--format",
        choices=["url", "b64_json", "b64_ephemeral"],
        default="url",
        help="Response format (default: url)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON response",
    )


def _add_models_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the models command."""
    parser.add_argument(
        "--type",
        choices=["video", "image"],
        help="Filter by output type",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )


def _add_credits_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the credits command."""
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )


def cmd_estimate(args: argparse.Namespace) -> int:
    """Handle the estimate command."""
    client = ImageRouterClient()
    estimator = CostEstimator(client)

    if args.type == "video":
        estimate = estimator.estimate_video(
            model=args.model,
            seconds=args.seconds,
            count=args.count,
        )
    else:
        estimate = estimator.estimate_image(
            model=args.model,
            quality=args.quality,
            size=args.size,
            count=args.count,
        )

    if args.json:
        print(json.dumps(estimate.to_dict(), indent=2))
    else:
        print(estimate.format_summary())

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle the generate command."""
    client = ImageRouterClient()

    if args.type == "video":
        generator = VideoGenerator(client)

        if args.image:
            # Image-to-video
            result = generator.image_to_video(
                image_path=args.image,
                prompt=args.prompt,
                model=args.model,
                seconds=args.seconds if args.seconds else "auto",
                size=args.size,
                response_format=args.format,
                output_path=args.output,
            )
        else:
            # Text-to-video
            result = generator.text_to_video(
                prompt=args.prompt,
                model=args.model,
                seconds=args.seconds if args.seconds else "auto",
                size=args.size,
                response_format=args.format,
                output_path=args.output,
            )
    else:
        generator = ImageGenerator(client)

        if args.image:
            # Image-to-image
            result = generator.image_to_image(
                image_path=args.image,
                prompt=args.prompt,
                model=args.model,
                mask_path=args.mask,
                quality=args.quality,
                size=args.size,
                response_format=args.format,
                output_path=args.output,
            )
        else:
            # Text-to-image
            result = generator.text_to_image(
                prompt=args.prompt,
                model=args.model,
                quality=args.quality,
                size=args.size,
                response_format=args.format,
                output_path=args.output,
            )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Print summary
        data = result.get("data", [])
        if data:
            for i, item in enumerate(data):
                if "url" in item:
                    print(f"Output {i + 1}: {item['url']}")
                elif "b64_json" in item:
                    print(f"Output {i + 1}: [base64 data, {len(item['b64_json'])} chars]")

        if args.output:
            print(f"Saved to: {args.output}")

    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Handle the models command."""
    client = ImageRouterClient()
    registry = ModelRegistry(client)

    models = registry.get_models_by_type(args.type) if args.type else registry.get_all_models()

    if args.json:
        output = {
            model_id: {
                "name": info.name,
                "provider": info.provider,
                "output_types": info.output_types,
                "pricing": {
                    "type": info.pricing.pricing_type,
                    "min": info.pricing.min_price,
                    "max": info.pricing.max_price,
                    "average": info.pricing.average_price,
                },
                "durations": info.supported_durations,
                "sizes": info.supported_sizes,
                "supports_edit": info.supports_edit,
            }
            for model_id, info in models.items()
        }
        print(json.dumps(output, indent=2))
    else:
        type_label = "video" if args.type == "video" else "image" if args.type == "image" else ""
        print(f"Available {type_label} models:\n")
        for model_id, info in sorted(models.items()):
            min_price, _, max_price = info.pricing.get_estimate()
            if min_price == max_price:
                price_str = f"${min_price:.2f}"
            else:
                price_str = f"${min_price:.2f} - ${max_price:.2f}"
            print(f"  {model_id}")
            print(f"    Provider: {info.provider}")
            print(f"    Price: {price_str}")
            if info.supported_durations:
                print(f"    Durations: {info.supported_durations}s")
            if info.supports_edit:
                print("    Supports edit: Yes")
            print()

    return 0


def cmd_credits(args: argparse.Namespace) -> int:
    """Handle the credits command."""
    client = ImageRouterClient()
    credits = client.get_credits()

    if args.json:
        print(json.dumps(credits, indent=2))
    else:
        remaining = credits.get("remaining_credits", 0)
        usage = credits.get("credit_usage", 0)
        deposits = credits.get("total_deposits", 0)

        print("Account Balance:")
        print(f"  Remaining credits: ${remaining:.2f}")
        print(f"  Total usage: ${usage:.2f}")
        print(f"  Total deposits: ${deposits:.2f}")

    return 0


def main() -> NoReturn:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "estimate":
            exit_code = cmd_estimate(args)
        elif args.command == "generate":
            exit_code = cmd_generate(args)
        elif args.command == "models":
            exit_code = cmd_models(args)
        elif args.command == "credits":
            exit_code = cmd_credits(args)
        else:
            parser.print_help()
            exit_code = 1

        sys.exit(exit_code)

    except ImageRouterError as e:
        print(f"Error: {e.message}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
