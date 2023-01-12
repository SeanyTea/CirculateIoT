import time
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import json
from datetime import datetime 

def helloworld(self,params,packet):
	print("Received Message")
	print("Payload:",packet.payload)
	payloadMsg = packet.payload
	print(payloadMsg['timestamp'])

myMQTTClient = AWSIoTMQTTClient("MacOS") #random key, if another connection using the same key is opened the previous one is auto closed by AWS IOT
myMQTTClient.configureEndpoint("a2rhn7hbr9ksxp-ats.iot.us-east-2.amazonaws.com", 8883)

myMQTTClient.configureCredentials("/Users/seanthammakhoune/Documents/AWS IoT Core/Mac/AmazonRootCA1_mac.pem", "/Users/seanthammakhoune/Documents/AWS IoT Core/Mac/private_mac.pem.key", "/Users/seanthammakhoune/Documents/AWS IoT Core/Mac/certificate_mac.pem.crt")

myMQTTClient.configureOfflinePublishQueueing(-1) # Infinite offline Publish queueing
myMQTTClient.configureDrainingFrequency(2) # Draining: 2 Hz
myMQTTClient.configureConnectDisconnectTimeout(10) # 10 sec
myMQTTClient.configureMQTTOperationTimeout(5) # 5 sec
print ('Initiating Realtime Data Transfer From Raspberry Pi...')
myMQTTClient.connect()

myMQTTClient.subscribe("MLX90640", 1, helloworld)
while True:
	

	time.sleep(.1)