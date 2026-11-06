from pathlib import Path
import re
import re
from file_manager import (
    TRAIN_DIR,
    recover_temp,
    temporarily_withhold,
    ROOT_DIR
)
import subprocess
import sys

# Display name -> original dataset filename prefix
TARGET_ATTACKS = {
    "DDoS-ICMP": "TCP_IP-DDoS-ICMP",
    "DDoS-SYN": "TCP_IP-DDoS-SYN",
    "DDoS-TCP": "TCP_IP-DDoS-TCP",
    "DDoS-UDP": "TCP_IP-DDoS-UDP",
    "DoS-UDP": "TCP_IP-DoS-UDP",
    "DoS-ICMP": "TCP_IP-DoS-ICMP",
    "DoS-SYN": "TCP_IP-DoS-SYN",
    "DoS-TCP": "TCP_IP-DoS-TCP",
    "MQTT-DDoS-Connect_Flood": "MQTT-DDoS-Connect_Flood",
    "Recon-Port_Scan": "Recon-Port_Scan",
}

def get_attack_files(prefix):
    """Find every training CSV belonging to an attack type."""
    pattern = re.compile(
        rf"^{re.escape(prefix)}\d*_train\.pcap\.csv$"
    )

    return sorted(
        file
        for file in TRAIN_DIR.glob("*.csv")
        if pattern.fullmatch(file.name)
    )


def get_available_attacks():
    """Associate the ten selected attacks with their CSV files."""
    return {
        name: get_attack_files(prefix)
        for name, prefix in TARGET_ATTACKS.items()
    }


def display_attacks(attacks):
    """Display numbered attack types and their CSV counts."""
    print("\nAvailable Attack Types")
    print("----------------------")

    for number, (name, files) in enumerate(attacks.items(), 1):
        status = f"{len(files)} CSV files"

        if not files:
            status = "NO FILES FOUND"

        print(f"[{number}] {name} ({status})")


def select_attack(attacks):
    """Let the user select an attack type by number."""
    display_attacks(attacks)

    choice = input("\nSelect attack number: ").strip()

    if not choice.isdigit():
        print("Please enter a number.")
        return None

    index = int(choice) - 1
    items = list(attacks.items())

    if not 0 <= index < len(items):
        print("Invalid selection.")
        return None

    attack_name, files = items[index]

    if not files:
        print(f"No CSV files found for {attack_name}.")
        return None

    print(f"\nSelected: {attack_name}")
    print(f"Files to withhold: {len(files)}")

    for file in files:
        print(f"  - {file.name}")

    return attack_name, files

def run_cnn(class_config=2, max_rows_per_file=1000):
    """Execute the original CNN training and evaluation script."""
    script_path = ROOT_DIR / "src" / "main.py"

    command = [
        sys.executable,
        str(script_path),
        "--class_config",
        str(class_config)
    ]

    if max_rows_per_file is not None:
        command.extend([
            "--max_rows_per_file",
            str(max_rows_per_file)
        ])

    print("\nStarting CNN training...", flush=True)

    subprocess.run(
        command,
        cwd=ROOT_DIR,
        check=True
    )

    print("\nCNN experiment completed.", flush=True)

def main():
    recover_temp()

    print("\nIoMT Unseen Attack Detection")
    print("===========================")

    while True:
        print("\n[1] List attack types")
        print("[2] Select attack to withhold")
        print("[0] Exit")

        command = input("\nEnter command: ").strip()

        if command == "0":
            print("Exiting.")
            break

        attacks = get_available_attacks()

        if command == "1":
            display_attacks(attacks)

        elif command == "2":
            selection = select_attack(attacks)

            if selection is not None:
                attack_name, files = selection

                confirm = input(
                    f"\nRun CNN with {attack_name} withheld? (y/n): "
                ).strip().lower()

                if confirm != "y":
                    print("Experiment cancelled.")
                    continue

                try:
                    with temporarily_withhold(files):
                        run_cnn(class_config=2)

                except subprocess.CalledProcessError as error:
                    print(
                        f"\nCNN training failed "
                        f"(exit code {error.returncode})."
                    )
                    print("Check the error output above.")

        else:
            print("Unknown command.")


if __name__ == "__main__":
    main()
