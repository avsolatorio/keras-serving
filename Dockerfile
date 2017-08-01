FROM tf-model-server

MAINTAINER Christian Fröhlingsdorf <chris@5cf.de>

RUN mkdir /model_export
ADD ./export /model_export/

CMD ["./tensorflow_model_server", "--port=9000", "--enable-batching", "--model_name=main_model", "--model_base_path=/model_export/main_model"]
