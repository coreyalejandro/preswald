from preswald.main import start_server
import os

script_path = os.environ.get('SCRIPT_PATH', 'hello.py')
port_str = os.environ.get('PORT', '8501')
try:
    port = int(port_str)
except (TypeError, ValueError):
    port = 8501

if __name__ == "__main__":
    # Launch the Preswald app with the configured script and port
    start_server(script=script_path, port=port)