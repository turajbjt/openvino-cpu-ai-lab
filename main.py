# main.py
import openvino.runtime as ov

def main():
    core = ov.Core()
    devices = core.available_devices
    
    print("\n--- OpenVINO Hardware Check ---")
    if devices:
        for device in devices:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"Detected Device: {device} ({device_name})")
    else:
        print("No OpenVINO devices found.")
    print("--------------------------------\n")
    print("OpenVINO Environment is Ready!")

if __name__ == "__main__":
    main()
