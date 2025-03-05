# Security Vision System

Esta aplicación web utiliza Flask, OpenCV, NumPy y YOLO para realizar detección de movimiento y alertar sobre posibles intrusos mediante un stream de video.

## Requisitos

- Python 3.x
- [Virtualenv](https://virtualenv.pypa.io/en/latest/)

## Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu máquina local:

1. **Clonar el repositorio**

   ```sh
   git clone https://github.com/Micky-CM/vision-system
   cd vision-system
   ```

2. **Crear y activar el entorno virtual**

- En Windows
  ```sh
  python -m venv venv
  venv\Scripts\activate
  ```

- En macOS/Linux
  ```sh
  python3 -m venv venv
  source venv/bin/activate
  ```

3. **Instalar las dependencias**

Una vez activado el entorno virtual, instala las dependencias con:
  ```sh
  pip install -r requirements.txt
  ```

4. **Ejecutar la aplicación**

Establece las variables de entorno y lanza la aplicación:
- En Windows
  ```sh
  set FLASK_APP=app.py
  set FLASK_ENV=development
  flask run
  ```

- En macOS/Linux
  ```sh
  export FLASK_APP=app.py
  export FLASK_ENV=development
  flask run
  ```

5. **Abrir la aplicación en el navegador**
Una vez ejecutado, abre tu navegador en http://127.0.0.1:5000 para ver la aplicación en funcionamiento.


## Estructura del Proyecto
```
/vision-system
├── app.py
├── models/
│   ├── yolov3.cfg
│   ├── yolov3.weights
│   └── coco.names
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── requirements.txt
└── README.md
```