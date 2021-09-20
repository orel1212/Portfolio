
from Crypto.PublicKey import RSA
from Crypto import Random
from Crypto.Signature.pkcs1_15 import PKCS115_SigScheme
from Crypto.Hash import SHA256
import socket
import sys
import json

def verify(publickey,data,signature):
	verifier = PKCS115_SigScheme(publickey)
	data_encoded = data.encode("utf8")
	hash = SHA256.new(data_encoded)
	try:
		verifier.verify(hash, signature)
		print("Verified Message: "+str(data)+" successfully!")
	except:
		print("Failed to verify Message: "+str(data))


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
ip = 'localhost'
port=10002

try:
	if len(sys.argv) > 1 :
		ip = str(sys.argv[1])
		if len(sys.argv) > 2 :
			port = int(sys.argv[2])
except Exception as e:
	print(e)
	exit()

server_address = (ip, port)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

f_k = open('pubkey.pem','r')
publickey=RSA.import_key(f_k.read())

while True:
	# Wait for a connection
	print('waiting for a connection')
	connection, client_address = sock.accept()
	try:
		print('connection from', client_address)
		# Receive the data in small chunks and retransmit it
		
		data = connection.recv(4096)
		if data:
			data=json.loads(data.decode())
			text=data.get("message")
			sign=bytes(bytearray(json.loads(data.get("signature"))))
			verify(publickey,text,sign)				
		else:
			print('no data recieved from', client_address)
			break

	
		connection.close()
	except Exception as e:
		print(e)
		# Clean up the connection
		connection.close()

	


