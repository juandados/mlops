FROM python:3.7
COPY . /mlops
WORKDIR mlops
RUN pip install pipenv
RUN pipenv install --dev --ignore-pipfile
