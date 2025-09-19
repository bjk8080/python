#파일을 읽어서 -> 암호화 -> 복호화
def xor_encrypt_decrypt(input_file,output_file,key):
try:
with open(input_file,'rb')as infile:
    data=infile.read
key_bytes = key.encode()if isinstance(key,str)else bytes

#key가 문자열이면 바이트 전환, 숫자면 바이트 리스트
encrypted_data=bytearray(len(data))
for i in range(len(data)):
    encrypted_data[i]=data[i] ^ key_bytes[i&key_len]
    #data =100, index :0~99
    #encrypted_data의 길이가 100
    #mykey123dml rlfdl 8,, index:0~7
    #주어진 index 90
    #90%8=2
    #91%8=3
with open(output_file, 'wb')as outfile:
    outfile.write(data)

except Exception as e:
    print(f'오류 {e}')

xor_encrypt_decrypt('example.txt', 'output.txt')
