# An End-to-End Self-driving Strategy

This repo contains code for implementing an end-to-end self-driving strategy through the conditional imitation learning 
algorithm (CIL). This algorithm takes photos shot from the front camera of vehicles and high-level command given by human as input
and make driving strategies including the throttle, brake and steer angle.
The CIL is trained under a Federated Learning framework.

To install the required dependencies, the following command can be run:

`pip install -r requirements.txt`