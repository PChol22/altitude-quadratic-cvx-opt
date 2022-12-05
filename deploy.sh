zip -r source.zip .
aws lambda update-function-code --function-name new-python-test --zip-file fileb://source.zip
