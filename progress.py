import os
import json
import time
import psutil
import sys
from datetime import datetime
from pathlib import Path
from termcolor import colored
import threading
import queue
import select

def find_training_processes():
    """Find all running train.py processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and 'train.py' in ' '.join(cmdline):
                    processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def format_time(seconds):
    """Format seconds into human-readable time."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def estimate_completion_time(position, total_size, elapsed_time):
    """Estimate total time required based on current progress."""
    if position == 0 or total_size == 0:
        return "Calculating..."
    
    rate = position / elapsed_time
    remaining_items = total_size - position
    estimated_remaining_time = remaining_items / rate if rate > 0 else 0
    
    return format_time(estimated_remaining_time)

def get_throttle_status(status):
    """Get colored throttle status message."""
    throttle_level = status.get('throttle_level', 0)
    reduced_batch = status.get('reduced_batch_size', False)
    
    if throttle_level == 2:
        return colored("SEVERE THROTTLING", "red")
    elif throttle_level == 1:
        return colored("SLIGHT THROTTLING", "yellow")
    elif reduced_batch:
        return colored("REDUCED BATCH SIZE", "yellow")
    else:
        return colored("ALL SYSTEMS GO", "green")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_status(status_file, command_queue):
    """Display training status."""
    while True:
        try:
            # Check for commands
            try:
                cmd = command_queue.get_nowait()
                if cmd == 'q':
                    break
            except queue.Empty:
                pass

            # Clear screen
            clear_screen()
            
            # Find training processes
            processes = find_training_processes()
            
            # Display number of processes
            print(colored(f"Number of Existing processes: {len(processes)}", "blue"))
            print()
            
            # Read and display status if available
            if os.path.exists(status_file):
                try:
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    # Calculate time estimates
                    timestamp = datetime.fromisoformat(status['timestamp'])
                    elapsed_time = (datetime.now() - timestamp).total_seconds()
                    eta = estimate_completion_time(
                        status['dataset_position'],
                        status['total_dataset_size'],
                        elapsed_time
                    )
                    
                    # Display status information
                    print(f"Status: {get_throttle_status(status)}")
                    print(f"Position in Dataset: {status['dataset_position']}/{status['total_dataset_size']}")
                    print(f"On batch: {status['batch_num']}/{status['total_batches']}")
                    
                    if status['latest_checkpoint']:
                        print(colored(f"Latest checkpoint: {status['latest_checkpoint']}", "green"))
                    
                    print(f"Current Loss: {status['loss']:.4f}")
                    print(f"CPU Usage: {status['cpu_percent']}%")
                    print(f"Memory Usage: {status['memory_percent']}%")
                    print(f"Estimated Time Remaining: {eta}")
                    
                except json.JSONDecodeError:
                    print(colored("Error reading status file", "red"))
                except Exception as e:
                    print(colored(f"Error: {str(e)}", "red"))
            else:
                print(colored("No active training session found", "yellow"))
            
            # Display commands
            print("\nCMD: (r=refresh, q=quit)")
            
            # Wait before next update
            time.sleep(1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(colored(f"Error: {str(e)}", "red"))
            time.sleep(1)

def input_thread(command_queue):
    """Handle keyboard input in a separate thread."""
    while True:
        if select.select([sys.stdin], [], [], 0.1)[0]:  # Check if input is available
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'q':
                command_queue.put('q')
                break
            elif cmd == 'r':
                continue  # Main loop automatically refreshes

def main():
    # Look for status file in current directory or output directory
    status_file = Path("output") / "training_status.json"
    
    if len(sys.argv) > 1:
        status_file = Path(sys.argv[1])
    
    # Create command queue for thread communication
    command_queue = queue.Queue()
    
    # Start input thread
    input_thread_obj = threading.Thread(target=input_thread, args=(command_queue,))
    input_thread_obj.daemon = True
    input_thread_obj.start()
    
    try:
        # Start display loop
        display_status(status_file, command_queue)
    except KeyboardInterrupt:
        print("\nExiting progress monitor...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        if input_thread_obj.is_alive():
            command_queue.put('q')
            input_thread_obj.join(timeout=1)

if __name__ == "__main__":
    main()
