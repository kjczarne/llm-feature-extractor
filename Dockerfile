FROM "python:3.10.13-alpine3.18"
RUN "curl -sSL https://install.python-poetry.org | python3 -"
CMD ["/bin/sh"]
