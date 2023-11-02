from greenness_track_toolkit import Server

if __name__ == '__main__':
    Server(server_mode="LOCAL", log_path="./log", api_port=8001).start()
