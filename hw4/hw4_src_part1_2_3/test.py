import matlab.engine


eng = matlab.engine.start_matlab()


print(eng.mytest('oja', 2, 50, ['./images/bike.jpg', './images/face.jpg'], 'all', [10, 2], 'corrupt', [1, 20, 10,10]))