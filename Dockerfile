FROM tensorflow/tensorflow

WORKDIR /app
ADD . /app
CMD cd /app && python celsius_a_fahrenheit.py && python multiplos_de_dos.py
