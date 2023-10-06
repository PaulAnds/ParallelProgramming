# Investigación CUDA Cores, Threads, Blocks and Grids

#### Programación de paralelismo
#### Paul Andres Solis Villanueva
#### University of Advanced Technologies
#### Leonardo Juárez Zucco
#### 06/10/2023


Núcleo Cuda- es una unidad de procesamiento fundamental de una GPU desarrollada por NVIDIA.
Son usados para usarse en tareas que necesitan procesamientos paralelos. Unos ejemplos son cálculos intensivos tipo renderización de gráficos en 3D, la inteligencia artificial y la computación científica. 

Funcion:
	La función de un núcleo CUDA es para cálculos paralelos, que significa, que se usan para ejecutar múltiples operaciones al mismo tiempo. Esto es fundamental para hacer tareas que tienen varios cálculos, como procesamiento de imágenes y simulaciones científicas.

Una GPC tiene miles de núcleos CUDA agrupados en Streaming Multiprocessors, cada SM tiene varios núcleos, igualmente que memoria compartida.

Procesamientos paralelos son procedimientos que se ejecutan al mismo tiempo haciéndolo más rápido y eficiente para calcular datos.

Tipos de núcleos CUDA:
	FP32- cálculos para flotantes
	FP64- cálculos para dobles

Se usa programación CUDA C/C + +, con un código que se ejecuta en la CPU que se dice por ser el “anfitrión” y otro código que se va a la GPU que es el “dispositivo”, y se especifica cómo se deben comunicar entre ellos por los datos.

Aplicaciones de los núcleos CUDA:
Videojuegos
Renderización 3D
Simulaciones científicas
Deep learning
Criptomineria
Procesamiento de imagenes medicas
Modelado climatico


cuDNN (CUDA Deep Neural Network) se usa para facilitar el desarrollo de Deep Learning

Jerarquía del paralelismo en CUDA:

La capa más alta son bloques de hilos y en la más baja están los hilos en esos bloques. Cada bloque de los hilos se ejecuta en un Streaming Multiprocessor (SM) 



Bloques:
Estos grupos de hilos se ejecutan independientemente y al mismo tiempo en un SM. Cada SM tiene una cantidad limitada de recursos, ejemplo - registros y la memoria compartida, que se comparte también entre los hilos dentro del bloque. El programador puede especificar el número de hilos por bloque y el número de bloques en una cuadrícula (grid) que ayuda a la optimización del rendimiento.

Hilos:
Los hilos individuales ejecutan instrucciones de manera simultánea. Los núcleos CUDA manejan múltiples hilos en un ciclo de reloj(muchos hilos avanzan en sus tareas al mismo tiempo). Esto es fundamental para acelerar los trabajos que tienen muchas operaciones aritméticas o operaciones lógicas.

Los hilos dentro de un bloque pueden cooperar y comunicarse utilizando la memoria compartida, es una memoria de acceso rápido que usan todos los hilos dentro del bloque. Esto ayuda a la sincronización y la colaboración entre ellos.


CPU - instrucciones secuencialmente en un solo núcleo
GPU - ejecutar millones de hilos simultáneamente en varios núcleos CUDA.

Bloques se ejecutan en un solo SM 

Los Grids pueden ser bidimensionales o tridimensionales. Filas y columnas (2d) o filas, columnas y capas (3d). Los bloques dentro de los grids se ejecutan al mismo tiempo, y si es necesario se pueden coordinar y comunicar entre sí con la memoria global compartida.






Referencias

Á. Aller, “Qué son Los Nvidia Cuda cores y cuál es su importancia,” Profesional Review, https://www.profesionalreview.com/2018/10/09/que-son-nvidia-cuda-core/  (accessed Oct. 5, 2023). 

Redacción, “Tu Gráfica Nvidia tiene cientos de ellos, pero ¿Qué son Los Núcleos Cuda?,” HardZone, https://hardzone.es/marcas/nvidia/nucleos-cuda/  (accessed Oct. 5, 2023). 

D. E. Alonso, “CITESEERX,” Intro to CUDA, https://citeseerx.ist.psu.edu/document?repid=rep1&amp;type=pdf&amp;doi=7647cd49b0226df1303dd0ff068820baf9c20fdd  (accessed Oct. 5, 2023). 

U. DE BURGOS, “Universidad de Burgos - Riubu Principal,” INTRODUCCIÓN A LA PROGRAMACIÓN EN CUDA,  https://riubu.ubu.es/bitstream/handle/10259/3933/Programacion_en_CUDA.pdf;jsessionid=504EBE42B44052D65E05101BEF76D724?sequence=1  (accessed Oct. 5, 2023). 

