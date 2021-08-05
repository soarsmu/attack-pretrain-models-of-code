FROM zhouyang996/codebert-attack:v1
WORKDIR /usr/hejunda
COPY . .
CMD ["test.py"]
ENTRYPOINT ["python3"]
