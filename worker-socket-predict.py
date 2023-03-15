from damast.ml.worker import Worker

if __name__ == "__main__":
    w = Worker()
    w.listen_and_accept()
