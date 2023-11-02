import argparse

from greenness_track_toolkit import Server


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_port", type=int, default=8000)
    parser.add_argument("--rpc_port", type=int, default=16886)
    parser.add_argument("--server_mode", type=str, default="SERVER", help="SERVER or LOCAL")
    parser.add_argument("--log_path", type=str, default="./", help="for server, it is log location,"
                                                                   "for local, it is log save path")
    return parser.parse_known_args()


def main():
    args, _ = arg_parse()
    Server(api_port=args.api_port, rpc_port=args.rpc_port, log_path=args.log_path, server_mode=args.server_mode).start()


if __name__ == '__main__':
    main()