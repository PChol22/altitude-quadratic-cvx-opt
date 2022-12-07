mv ./src/lambda_function.py ./lambda_function.py
zip -r source.zip ./cvxopt ./cvxopt.libs ./numpy ./numpy.libs ./pytz ./lambda_function.py
mv ./lambda_function.py ./src/lambda_function.py
aws lambda update-function-code --function-name new-python-test --zip-file fileb://source.zip
