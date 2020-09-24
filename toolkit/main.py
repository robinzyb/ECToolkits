import argparse
import sys
from toolkit.analysis.analysis import analysis_run

def main():
    print("Description\n------------")
    parser = argparse.ArgumentParser(description="""
    reweight is a convenient script to reweight the quantity from commitee
    potential, to view the sub-command help,  type "reweight sub-command -h.
    the potentials stored must be eV unit""")

    subparsers = parser.add_subparsers()
    # run
    parser_run = subparsers.add_parser(
            "run", help="main function to reweight the quantity")
    parser_run.add_argument('CONFIG', type=str,
            help="parameter file, json format")
    parser_run.set_defaults(func=analysis_run)




    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(0)

    args.func(args)

if __name__ == "__main__":
    main()
