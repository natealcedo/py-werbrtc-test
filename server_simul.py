import asyncio
import websockets
import json
import time
import logging
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import os
import pygame
import sys
import math

# This is mandatory !!! if you keep ProactorEventLoop policy keyboard input won't work
# correctly with the Websocket server asyncio's event handling under Windows (I tested under Win11)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'

    # logs are being sent to display. If you need to send them to a disk file
    # uncomment and adjust following lines
    # filename='server.log',  # Logs will be saved to this file
    # filemode='a'  # 'a' for append, 'w' for overwrite
)


class WebSocketServer:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        logging.info(f"Initializing WebSocket server on {host}:{port}")

    async def handler(self, websocket):
        self.clients.add(websocket)
        logging.info(f"New client connected")
        try:
            async for message in websocket:
                logging.info(f"Received message: {message}")
        except websockets.ConnectionClosed:
            logging.warning("Connection closed")
        finally:
            self.clients.remove(websocket)

    async def send_message(self, message):
        if not self.clients:
            return
        tasks = [asyncio.create_task(client.send(message)) for client in self.clients]
        try:
            await asyncio.gather(*tasks)
            logging.debug(f"Sent message to {len(self.clients)} clients: {message}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")

    async def start(self):
        try:
            server = await websockets.serve(lambda ws: self.handler(ws), self.host, self.port)
            logging.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            return server
        except Exception as e:
            logging.error(f"Failed to start server on {self.host}:{self.port}: {e}")
            raise


class UxInputCommandHandler(ABC):
    @abstractmethod
    async def get_command(self):
        pass


class KeyboardCommandHandler(UxInputCommandHandler):
    async def get_command(self):
        await asyncio.sleep(0.1)
        return None


class JoystickCommandHandler(UxInputCommandHandler):
    def __init__(self):
        try:
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise Exception("No joystick found")
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Joystick initialized")
        except Exception as e:
            logging.error(f"Joystick initialization failed: {e}")
            raise

    async def get_command(self):
        pygame.event.pump()
        if self.joystick.get_button(5):  # Right bumper
            return 'n'
        if self.joystick.get_button(4):  # Left bumper
            return 'p'
        if self.joystick.get_button(0):  # A button
            return 'c'
        await asyncio.sleep(0.1)
        return None


class SensorDataHandler:
    def __init__(self, delay_ms):
        self.delay_seconds = delay_ms / 1000.0
        self.base_temperature = 25.5
        self.time = 0
        self.battery_level = 100
        self.battery_update_counter = 0  # Counter for battery updates
        self.battery_update_threshold = int(2000 / delay_ms)  # Number of iterations for 2 seconds

    async def send_sensor_data(self, websocket_server):
        while True:
            self.time += 1
            temp_variation = math.sin(self.time * 0.1) * 2
            acc_x_variation = math.sin(self.time * 0.2) * 0.1
            acc_y_variation = math.cos(self.time * 0.2) * 0.1
            acc_z_variation = math.sin(self.time * 0.1) * 0.05

            # Update battery level every 2 seconds
            self.battery_update_counter += 1
            if self.battery_update_counter >= self.battery_update_threshold:
                self.battery_update_counter = 0
                self.battery_level -= 1
                if self.battery_level < 0:
                    self.battery_level = 100

            data = {
                "accelerometer": {
                    "x": 0.05 + acc_x_variation,
                    "y": -0.02 + acc_y_variation,
                    "z": 9.81 + acc_z_variation
                },
                "temperature": self.base_temperature + temp_variation,
                "timestamp": datetime.now(timezone.utc).isoformat() + 'Z',
                "battery": self.battery_level
            }
            json_data = json.dumps(data, indent=2)
            print("\nSending telemetry data:")
            print(json_data)
            await websocket_server.send_message(json_data)
            await asyncio.sleep(self.delay_seconds)


def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Loaded config: {config}")
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        raise


async def handle_ux_input_commands(ux_websocket_server, ux_input_command_handler):
    while True:
        command = await ux_input_command_handler.get_command()
        if command is None:
            continue
        timestamp = int(time.time())
        if command.lower() == 'c':
            message = {"command": "ux_capture_image", "timestamp": timestamp}
        elif command.lower() == 'p':
            message = {"command": "ux_previous", "timestamp": timestamp}
        elif command.lower() == 'n':
            message = {"command": "ux_next", "timestamp": timestamp}
        else:
            continue

        json_message = json.dumps(message, indent=2)
        print("\nSending UX command:")
        print(json_message)
        await ux_websocket_server.send_message(json_message)


async def main():
    try:
        logging.info("Loading configuration...")
        config = load_config('config.json')

        ux_server_config = config['ux_server']
        telemetry_server_config = config['telemetry_server']
        input_type = config['input_type']
        telemetry_delay = config.get('telemetry_delay_ms', 500)

        logging.info("Starting UX WebSocket server...")
        ux_websocket_server = WebSocketServer(
            host=ux_server_config['host'],
            port=ux_server_config['port']
        )

        logging.info("Starting Telemetry WebSocket server...")
        telemetry_websocket_server = WebSocketServer(
            host=telemetry_server_config['host'],
            port=telemetry_server_config['port']
        )

        sensor_data_handler = SensorDataHandler(telemetry_delay)

        if input_type == 'keyboard':
            ux_input_command_handler = KeyboardCommandHandler()
        elif input_type == 'joystick':
            ux_input_command_handler = JoystickCommandHandler()
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        logging.info("Starting all services...")
        ux_server = await ux_websocket_server.start()
        telemetry_server = await telemetry_websocket_server.start()

        await asyncio.gather(
            sensor_data_handler.send_sensor_data(telemetry_websocket_server),
            handle_ux_input_commands(ux_websocket_server, ux_input_command_handler)
        )

        await ux_server.wait_closed()
        await telemetry_server.wait_closed()

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
    except Exception as e:
        logging.error(f"Server failed to start: {e}")
        sys.exit(1)