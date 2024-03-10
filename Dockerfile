FROM python:3.8

WORKDIR /app

VOLUME /models

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Копирование файлов приложения в контейнер
COPY train_model.py /app/train_model.py
COPY test_model.py /app/test_model.py
COPY train_df.csv /app/train_df.csv
COPY test_df.csv /app/test_df.csv

ENV TASK=train

CMD if [ "$TASK" = "train" ]; then \
        python train_model.py; \
    elif [ "$TASK" = "test" ]; then \
        python test_model.py; \
    else \
        echo "Неправильно указана задача. Используйте 'train' или 'test'."; \
    fi