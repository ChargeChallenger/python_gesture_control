Курсовая работа "Управление персональным компьютером при помощи жестов"

Эта программа позволяет изменять громкость звука на персональном компьютере, ставить видео на паузу
при помощи обученной нейросети. Описанные действия программа выполняет, нажимая хоткеи.
Данная программа использует уже готовую нейросеть, найденную в сети, определяющую один из 4 жестов:
1. Fist (кулак) - Убавить звук
2. Point (Уазательный палец вверх) - Прибавить звук
3. Five (Пятерня) - Выключить звук
4. Swing (Шака) - Пауза/Воспроизведение


Используемые библиотеки:
keras
numpy
cv2
pyautogui (Установка: "pip install pyautogui")


Требования:
1. Веб-камера
2. Хорошо освещённая комната


Запуск и настройка:
Запускается программа через gesturecontrol.py
Программа первоначально запоминает первый кадр как задний фон, используя для последующего сравнения кадров
Если в открывшихся окнах dilation и hand_crop всё белым-бело, то фон определился некорректно.
Рекомендуется включить свет, сесть немного дальше от веб-камеры, разместившись слева.
Затем нажмите клавишу "R", чтобы переопределить задний фон
Если возникают проблемы - перезапустите программу

Синий квадрат на foreground_display - поле определения жестов. Для определения одного из четырёх жестов
поместите руку в квадрат. Если на foreground_display написано "hand pose: none", то переместите руку ближе
к камере, так как рука занимала малый объём кадра.

Изначально в программе отключено управление компьютером, чтобы не возникало дополнительных проблем
при определении заднего фона. После настройки (на экранах dilation и hand_crop должен быть черный фон 
и отображаться движения руки белым цветом) нажмите Пробел, чтобы включить управление. Повторное нажатие
выключает управление

Нажмите ESC, чтобы выключить программу.

