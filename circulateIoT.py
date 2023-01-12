from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import time
import json
def helloworld(self,params,packets):
    print("Received message from IoT core")
    print("Topic:", packets.topic)
    print("Payload:", packets.payload)
# For certificate based connection
myMQTTClient = AWSIoTMQTTClient("test")
# For TLS mutual authentication
myMQTTClient.configureEndpoint("a2rhn7hbr9ksxp-ats.iot.us-east-1.amazonaws.com", 8883) #Provide your AWS IoT Core endpoint (Example: "abcdef12345-ats.iot.us-east-1.amazonaws.com")
myMQTTClient.configureCredentials("/home/intelligentmedicine/Desktop/AWS_IoT/AmazonRootCA1.pem", "/home/intelligentmedicine/Desktop/AWS_IoT/private_RPI.pem.key", "/home/intelligentmedicine/Desktop/AWS_IoT/certificate_RPI.pem.crt") #Set path for Root CA and provisioning claim credentials
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(10)
myMQTTClient.configureMQTTOperationTimeout(5)

myMQTTClient.connect()
start_time = time.time()
while time.time() - start_time < 5:
    payloadMsg = {}
    payloadMsg['timestamp'] = time.time()
    payloadMsg['text'] = 'Hello World!'
    payloadMsg = json.dumps(payloadMsg)
    myMQTTClient.publish(topic = "helloWorld", QoS=1, payload = payloadMsg)
    time.sleep(1)
myMQTTClient.disconnect()
