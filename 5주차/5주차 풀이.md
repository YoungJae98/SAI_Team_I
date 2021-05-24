# 5주차 과제

# 문제 1: 이론

- 문제 1번

    ## 문제타입

    - [x]  이론
    - [ ]  실습

    ---

    ## 문제내용

    다음 문제를 보고 옳은 것은 O, 틀린 것은 X로 답하시오. (각 문제당 2점)

    ```
    (1) Overfitting 상태에서는 test set의 손실보다 training set의 손실이 더 적다.
    (2) Training set을 늘리면 overfitting 문제를 해결할 수 있다.
    (3) Training set은 모델의 성능을 측정하기 위해 사용되며, Validation set은 모델을 학습시키는데 사용된다.
    (4) Test set은 최종 모델의 성능을 측정하기 위해 여러 번 사용할 수 있다.
    (5) Training set의 정확도가 99%이고 Validation set의 정확도가 80%라면 Overfitting을 의심할 수 있다.

    (1) :
    (2) :
    (3) :
    (4) :
    (5) :
    ```

    ---

    ## 출제 정보

    ### 출제자

    출제자 1 : 조용재 (6팀)- 1,2 출제

    출제자 2 : 하린 - 3,4 출제

    ### 검수자

    검수자1 : 배정준 - 기존 5번 내용에 모호한 부분이 있어서 교체

    검수자2 : 권동하

    최종 검수자 : 전원

    ### 출제의도

    Overfitting을 이해했고 Training set, Validation set, Test set을 구분할 수 있는지 확인

- 풀이 1번

    답: 
    (1) : O
    (2) : O
    (3) : X (Training set은 모델을 학습시키는데 사용되며,Validation set은 모델의 성능을 측정하기 위해 사용된다.)
    (4) : X (Test set은 오직 한번만 사용한다.)
    (5) : O

    ### 기여자

    모든 팀원이 검토하며 문제를 풀었다.

    강찬울, 양가영, 이연경 - 2,4번 개념을 잡아주었다.

    ### 검수자

    이연경

    ### 참고한 자료

    [Training, Validation and Test sets 차이 및 정확한 용도 (훈련, 검정, 테스트 데이터 차이)](https://modern-manual.tistory.com/19)

---

# 문제 2: 데이터를 다뤄봅시다.

- 문제 2번

    ## 문제타입

    - [ ]  이론
    - [x]  실습

    ---

    ## 문제내용

    문제 내용

    강의에서는 tensorflow에서 준비해준 MNIST  Dataset을 이용했습니다.
    그런데..!? tensorflow에서 내가 사용하고 싶은 Dataset을 제공하지 않으면 어떡하죠?
    아래에 제공되는 MNIST 데이터를 다운로드해서 아래 코드를 완성해주세요!

    [mnist.pkl.gz](https://s3.amazonaws.com/img-datasets/mnist.pkl.gz)

    (압축 풀지말고 그대로 올려주세요!)

    ---

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a.png)

                       [꼭 업로드 완료 후 코드를 실행해주세요]

    ```python
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    import gzip
    import pickle

    def load_data(path):               
        f = gzip.open(path, 'rb')
        training_data, test_data = pickle.load(f, encoding = 'latin1')   
        f.close()
        return (training_data, test_data)

    (x_train, y_train), (x_test, y_test) = load_data("              ") # 파일 경로
    ```

    기쁜 마음으로 데이터를 로드하는데 성공했습니다. 
    강의에서는 test set의 shape가 아래와 같이 출력되었습니다.

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_shape.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_shape.png)

                           [강의코드의 shape]

    우리가 로드한 데이터도 같은 shape로 나올까요?

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_shape%201.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_shape%201.png)

      [우리가 로드한 데이터의 shape]

    이런 !! shape를 강의코드와 같게 변환해줘야 할 것 같아요.

    ```python
    def get_one_hot(targets, nb_classes):   # numpy 원 핫 인코딩을 구하는 함수
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    print("----before reshape----")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    nb_classes =                                   # 출력 개수

    x_train = x_train.reshape(       )             # 훈련 데이터
    y_train = get_one_hot(              )          # 훈련 타깃
    x_test = x_test.reshape(       )               # 테스트 데이터
    y_test = get_one_hot(              )           # 테스트 타깃

    print("----after reshape----")

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    ```

    shape를 강의코드와 동일하게 올바르게 변환했다면 한 가지 체크해야 할 것이 있어요.
    바로 데이터 Preprocessing 여부에요.
    강의코드의 데이터는 0~1 사이의 값으로 Preprocessing이 되어있어요.

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_.png)

                         [강의 코드의 데이터]

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_%201.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/_%201.png)

                        [우리가 로드한 데이터]

    ```python
    # preprocessing은 도와드릴게요
    x_train, x_test = x_train/255, x_test/255
    ```

    ```python
    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    W = tf.Variable(tf.random_normal([784, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))

    # Hypothesis (using softmax)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis= ))	# tf.reduce_sum의 axis 값이 의미하는 것은 무엇일까요?
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=    ).minimize(cost) # 최종 Accuracy 0.8 이상이 나오도록 learning_rate 찾아주세요
    											
    # Test model
    is_correct = tf.equal(tf.argmax(hypothesis,  ), tf.argmax(Y,  )) # tf.argmax 2번째 인수 값이 의미하는 것은 무엇일까요?
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    ```

    ```python
    	# numpy 미니 배치 제너레이터
    def gen_batch(x, y, batch_size):
        length = len(x)
        bins = length // batch_size # 미니배치 횟수
        if length % batch_size:
            bins += 1                    # 나누어 떨어지지 않을 때
        indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = batch_size * i
            end = batch_size * (i + 1)
            yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.

    # parameters
    training_epochs = 15
    batch_size = 100

    # with tf.Session() as sess:
    sess = tf.Session()
    # Initialize TensorFlow variables    
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):    
        avg_cost = 0    
        total_batch = len(x_train) // batch_size
        
        for batch_xs, batch_ys in gen_batch(              , batch_size): # 어떤 데이터를 학습에 이용해야 할까요?
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch         
        
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test, Y: y_test}))
    ```

    ```python
    import matplotlib.pyplot as plt
    import random

    # Get one and predict
    r = random.randint(0, len(x_test) - 1)
    # print(mnist.test.num_examples)
    print("Label:", sess.run(tf.argmax(y_test[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), 
                          feed_dict={X: x_test[r:r + 1]}))

    plt.imshow(x_test[r:r + 1].
              reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
    ```

    아래 2개 결과 출력을 제출해주세요!

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a%201.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a%201.png)

    ## 출제 정보

    ### 출제자

    출제자 1 : 권동하 : base 코드, numpy 원 핫 인코딩 함수

    출제자 2 : 배정준 : 데이터 로드, 미니 배치 함수 

    ### 검수자

    검수자1 : 조용재(6팀) : arg_max 함수 → argmax 함수 (arg_max 함수는 곧 deprecate 됨)

    검수자2 : 하린 : 압축파일관련 내용 추가

    최종 검수자 : 전원

    ### 출제의도

    Tensorflow에서 제공하지 않는 데이터를 다루는 방법을 확인해 보자.

- 풀이 2번

    ```python
    import numpy as np
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    import gzip
    import pickle

    def load_data(path):               
        f = gzip.open(path, 'rb')
        training_data, test_data = pickle.load(f, encoding = 'latin1')   
        f.close()
        return (training_data, test_data)

    (x_train, y_train), (x_test, y_test) = load_data("/content/mnist.pkl.gz") # 파일 경로

    def get_one_hot(targets, nb_classes):   # numpy 원 핫 인코딩을 구하는 함수
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    print("----before reshape----")
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    nb_classes = 10                                     # 출력 개수

    x_train = x_train.reshape(60000, 784)             # 훈련 데이터
    y_train = get_one_hot(y_train, nb_classes)          # 훈련 타깃
    x_test = x_test.reshape(10000, 784)               # 테스트 데이터
    y_test = get_one_hot(y_test, nb_classes)            # 테스트 타깃

    print("----after reshape----")

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # preprocessing은 도와드릴게요
    x_train, x_test = x_train/255, x_test/255

    # MNIST data image of shape 28 * 28 = 784
    X = tf.placeholder(tf.float32, [None, 784])
    # 0 - 9 digits recognition = 10 classes
    Y = tf.placeholder(tf.float32, [None, nb_classes])

    W = tf.Variable(tf.random_normal([784, nb_classes]))
    b = tf.Variable(tf.random_normal([nb_classes]))

    # Hypothesis (using softmax)
    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis= 1))	# tf.reduce_sum의 axis 값이 의미하는 것은 무엇일까요?
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # 최종 Accuracy 0.8 이상이 나오도록 learning_rate 찾아주세요
    											
    # Test model
    is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1)) # tf.argmax 2번째 인수 값이 의미하는 것은 무엇일까요?
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    	# numpy 미니 배치 제너레이터
    def gen_batch(x, y, batch_size):
        length = len(x)
        bins = length // batch_size # 미니배치 횟수
        if length % batch_size:
            bins += 1                    # 나누어 떨어지지 않을 때
        indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞습니다.
        x = x[indexes]
        y = y[indexes]
        for i in range(bins):
            start = batch_size * i
            end = batch_size * (i + 1)
            yield x[start:end], y[start:end]   # batch_size만큼 슬라이싱하여 반환합니다.

    # parameters
    training_epochs = 15
    batch_size = 100

    # with tf.Session() as sess:
    sess = tf.Session()
    # Initialize TensorFlow variables    
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):    
        avg_cost = 0    
        total_batch = len(x_train) // batch_size
        
        for batch_xs, batch_ys in gen_batch(x_train, y_train, batch_size): # 어떤 데이터를 학습에 이용해야 할까요?
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch         
        
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test, Y: y_test}))
    ```

    ---

    **<출력 결과>**

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled.png)

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%201.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%201.png)

    ### 기여자

    모든 팀원이 검토하며 문제를 풀었다.

    ### 검수자

    이연경

---

# 문제 3: 이론

---

- 문제 3번

    ## 문제타입

    - [x]  이론
    - [ ]  실습

    ---

    ## 문제내용

    다음 중 Learning rate와 Overfitting, Regularization에 대해서 올바르지 못한 것을 모두 고르시오. 

    < 그림 1 >

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%202.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%202.png)

    ```
    a. 위 <그림1>에서 오른쪽으로 변화하려면 regularization strength의 값을 작게 변화해야 한다. 
    b. Standardization은 결과에 대한 Feature의 영향력을 고루 분배하기 위해 사용하며, Normalization은 Outlier Treatment를 위해 사용한다.
    c. overfitting을 개선하는 과정에서 training data에 대한 모델의 정확도는 높아진다. 
    d. learning rate가 과도하게 크면 학습시 Cost의 값이 Nan으로 표현될 수 있다. 반대로 learning rate가 과도하게 작으면 일정 Cost에서 머무르는 현상이 나타나기도 한다.
    ```

    ---

    ## 출제 정보

    ### 출제자

    출제자: 김지현, 이예진, 조용재(7팀)

    ### 검수자

    최종 검수자: 7팀 전원

    ### 기타

    출제팀: 인공지능을 물리7팀

    출제일: 2021-05-19

- 풀이 3번

    답:  b, c
    이유
    b : standardization은 데이터가 얼마나 떨어져있는지를 나타내며 특정범위를 벗어나면 이상치로 간주하여 제거하고, normalization은 데이터의 상대적 크기에 대한 영향을 줄이기 위해 데이터들의 범위를 0~1로 변환한다. outlier treatment가 아니다.
    c : training accuracy는 높은데 validation accuracy는 낮은 경우 , training accuracy 희생해 validation accuracy와 맞춰줄 필요가 있다.

    ### 기여자

    모든 팀원이 검토하며 문제를 풀었다.

    ### 검수자

    이연경

# **문제 4:** 표준화와 정규화를 접해봅시다.

---

- 문제 4번

    ## 문제타입

    - [ ]  이론
    - [x]  실습

    ---

    ## 문제내용

    주어진 데이터의 각 feature을 표준화하여 z-score 절댓값 2이상을 이상치로하여 데이터를 정제하고, 이 데이터를 0~1로 정규화하여라.

    [iris.csv](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/iris.csv)

    ```python
    ##조건
    #standardization은 scipy의 zscore함수 이용.
    #normalization은 scipy의 MinMaxScaler함수 이용.
    #결과 : 소스코드, 결과 csv파일(첨부해주세요)
    import pandas as pd
    df = pd.read_csv()
    #=======Standardization을 진행해주세요(각 feature마다 Standardization이 진행되어야합니다.)=======

    #=======Normalization을 진행해주세요(각 feature마다 Normalization이 진행되어야합니다.)=======

    #=======결과 DataFrame을 csv로 저장해주세요=======
    ```

    ---

    ## 출제 정보

    ### 출제자

    출제자: 조용재(7팀)

    ### 검수자

    최종 검수자: 7팀 전원

    ### 기타

    출제팀: 인공지능을 물리7팀

    출제일: 2021-05-19

- 풀이 4번

    ```python
    ##조건
    #standardization은 scipy의 zscore함수 이용.
    #normalization은 scipy의 MinMaxScaler함수 이용.
    #결과 : 소스코드, 결과 csv파일(첨부해주세요)
    import pandas as pd
    from scipy.stats import zscore

    #scipy에는 minmaxscaler가 있는것을 확인하지 못해서 sklearn의 minmaxscaler를 사용함.
    from sklearn.preprocessing import MinMaxScaler
    df = pd.read_csv("iris.csv")
    #=======Standardization을 진행해주세요(각 feature마다 Standardization이 진행되어야합니다.)=======
    #번호 지우기
    new_df = df.iloc[:,1:5]

    df_sd = zscore(new_df, axis = 1)

    # z-score이 절댓값 2 이상이라면 이상치로 처리하기
    for i in range(150):
      for j in range(4):
        if(abs(df_sd[i,j]) >= 2):df_sd[i,j]=None

    #=======Normalization을 진행해주세요(각 feature마다 Normalization이 진행되어야합니다.)=======
    #scipy에 없기에 함수 구성, 본 코드는 sklearn
    #def MinMaxScaler(data):
    #    numerator = data - np.min(data, 0)
    #    denominator = np.max(data, 0) - np.min(data, 0)
    #    return numerator / (denominator + 1e-5)

    mm = MinMaxScaler()
    df_mm = mm.fit_transform(df_sd)
    #=======결과 DataFrame을 csv로 저장해주세요=======

    df_sd = pd.DataFrame(df_sd)
    df_mm = pd.DataFrame(df_mm)

    df_sd.to_csv("iris_standard.csv")
    df_mm.to_csv("iris_normalize.csv")
    ```

    ---

    **<출력 결과>**

    [iris_normalize.csv](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/iris_normalize.csv)

    [iris_standard.csv](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/iris_standard.csv)

    ### 기여자

    모든 팀원이 검토하며 문제를 풀었다.

    이연경 - 데이터중에서 무의미한 번호를 지우고 zscore에서 이상치를 처리하는점을 짚어주었다.

    양가영 - scipy에는 MinMaxScaler함수가 없어 이를 직접 함수로 구현하였다.

    ### 검수자

    이연경

---

# 문제 5: 데이터 전처리를 해보아요

---

- 문제 5번

    ## 문제타입

    - [ ]  이론
    - [x]  실습

    ---

    ## 문제내용

    [test.csv](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/test.csv)

    타이타닉호에 탑승한 승객들의 정보를 이용해 승객들의 생존 여부를 예측하려 합니다.  각각의 정보들에 대한 평균값으로 결측값을 채우고 정확한 예측을 위해 편차가 큰 값들을 0과 1 사이의 숫자로 바꿔주세요. (전처리는 'Pclass', 'Sex', 'Age', 'Fare' 네 개의 정보들만 진행합니다.)

    ```
    Age의 결측값은 Name에 적힌 정보들을 기반으로 유추하고자 합니다. 
    (Name에 적힌 신분과 Age가 연관성이 있을 것으로 예상되기 때문입니다.)

    Age의 결측값을 Name에 적힌 정보들의 평균으로 각각 값을 채워봅시다.
    (Name에 Mr가 포함된 행의 결측값은 Mr끼리의 평균, Mrs가 포함된 행의 결측값은 Mrs끼리의 평균)
    ```

    ![5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%203.png](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/Untitled%203.png)

    ▲ ex) 413번째 인덱스 값에서 Age 가 결측값이다. Name에 Mr가 포함되어 있으니까 다른 행들 중 Mr를 포함한 행들의 Age 값 평균으로 결측값을 대체한다.

    ```python
    import pandas as pd

    df = [             ] # csv 파일 불러오기

    # info 함수를 이용해 데이터 정보 확인 후 결측값을 채워주세요
    # 참고
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    ##############################################

    ##############################################

    df = [             ] # Pclass, Sex, Age, Fare 열 추출

    # replace 함수를 이용해 female, male을 각각 정수 0과 1로 치환해주세요
    ##############################################

    ##############################################

    # Pclass, Age, Fare의 값을 모두 각각의 최댓값으로 나누어 준 뒤
    # Pclass와 Age는 소수점 첫째자리까지 반올림
    # Fare는 소수점 셋째자리까지 반올림해주세요
    ##############################################

    ##############################################

    # 수정된 데이터프레임을 csv로 저장하여 올려주세요

    ```

    ## 완성된 csv파일

    - **힌트1**

        정규식을 이용해서 Name에 적힌 신분을 파악해 보세요...

    - **힌트2**

        Ms가 포함된 행은 하나뿐이기 때문에 평균으로 채울 수 없습니다. 

        Miss의 평균으로 채워주세요

    ---

    ## 출제 정보

    ### 출제자

    출제자: 유정하

    ### 검수자

    최종 검수자: 7팀 전원

    ### 기타

    출제팀: 인공지능을 물리7팀

    출제일: 2021-05-19

- 풀이 5번

    ```python
    import pandas as pd
    df = pd.read_csv("test.csv") # csv 파일 불러오기

    # info 함수를 이용해 데이터 정보 확인 후 결측값을 채워주세요
    print(df.info())
    #이를 통해서 age 86개, fare 1개의 결측값 확인

    ##############################################
    # 참고 => Fare의 결측값 하나 해결
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    #신분별로 구분하여 age 평균내기

    #새로 값을 옮기기(신분별로 구분해서)
    #결측값 데이터 확인해보니 Miss, Ms, Master에도 결측값이 존재하여 구분함.
    #Mr만 추가하는 str을 모을 경우 Mrs와 겹칠 수 있어 정규식을 사용
    new_mr = df[df['Name'].str.contains('Mr\.')]
    new_mrs = df[df['Name'].str.contains('Mrs\.')]
    new_ms = df[df['Name'].str.contains('Ms\.')]
    new_mas = df[df['Name'].str.contains('Master\.')]

    #옮긴값을 이용해서 결측값 채우기
    #Ms는 결측값이 하나여서 Miss를 사용하라는 힌트를 참고
    #Mr과 Mrs를 구분하기 위해 .을 추가
    for i in range(418):
      if (pd.isnull(df['Age'][i]) == True) & ('Mr.' in df['Name'][i]):
        df['Age'].fillna(new_mr['Age'].mean(),inplace=True)
      elif (pd.isnull(df['Age'][i]) == True) & ('Mrs.' in df['Name'][i]):
        df['Age'].fillna(new_mrs['Age'].mean(),inplace=True)
      elif (pd.isnull(df['Age'][i]) == True) & ('Miss.' in df['Name'][i]):
        df['Age'].fillna(new_miss['Age'].mean(),inplace=True)
      elif (pd.isnull(df['Age'][i]) == True) & ('Ms.' in df['Name'][i]):
        df['Age'].fillna(new_miss['Age'].mean(),inplace=True)
      elif (pd.isnull(df['Age'][i]) == True) & ('Master.' in df['Name'][i]):
        df['Age'].fillna(new_mas['Age'].mean(),inplace=True)

        
    #결측값 재확인
    print(df.info())

    ##############################################
    #df = [             ] # Pclass, Sex, Age, Fare 열 추출
    df = df[['Pclass','Sex','Age','Fare']]

    # replace 함수를 이용해 female, male을 각각 정수 0과 1로 치환해주세요
    ##############################################
    df['Sex'] = df['Sex'].replace("male", 1)
    df['Sex'] = df['Sex'].replace("female", 0)
    ##############################################

    # Pclass, Age, Fare의 값을 모두 각각의 최댓값으로 나누어 준 뒤
    # Pclass와 Age는 소수점 첫째자리까지 반올림
    # Fare는 소수점 셋째자리까지 반올림해주세요
    ##############################################
    size = df.shape[0] #len 함수 사용해도 됨
    max_Pclass = max(df['Pclass'])
    max_Age = max(df['Age'])
    max_Fare = max(df['Fare'])

    df['Pclass'] = round(df['Pclass']/max_Pclass, 1)
    df['Age'] = round(df['Age']/max_Age, 1)
    df['Fare'] = round(df['Fare']/max_Fare, 3)

    ##############################################
    df.to_csv("new_df.csv")
    # 수정된 데이터프레임을 csv로 저장하여 올려주세요
    ```

    ---

    **<출력 결과>**

    [new_df.csv](5%E1%84%8C%E1%85%AE%E1%84%8E%E1%85%A1%20%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6%20b84fc37bfdb44522aa5deea55cc60c6a/new_df.csv)

    ### 기여자

    김영재 - 데이터중에서 Mr, Mrs를 제외하고도 Miss, Ms, Master에 결측값이 있는것을 확인했다.

    이연경 - age에 대해서 신분별로 결측값을 채우는 간단한 방식을 제시했다.

    양가영 - 최댓값으로 나눈 후 반올림을 할때, 반복문을 이용하지 않아도 구현이 가능함을 알려주었다.

    ### 검수자

    이연경

---