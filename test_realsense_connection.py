import pyrealsense2 as rs2
import time

current_devices = []

def on_devices_changed(info):
    global current_devices
    devs = info.get_devices()
    print("on_devices_changed called new_devices:", devs.size())
    for dev in devs:
        current_devices.append(dev)

    for dev in current_devices:
        print("current:", dev, " added:", info.was_added(dev), " removed:", info.was_removed(dev))

def main():
    while True:
    ctx = rs2.context()
    ctx.set_devices_changed_callback(on_devices_changed)

if __name__ == '__main__':
    main()