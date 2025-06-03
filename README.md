Follow below instructions to check mono repo architecture:

1. Move to specific AI feature folder 
2. For text-emotion:
  Use below commands incase if you want to work with virtual environment: 
  # For text-emotion
  cd packages/text-emotion
  conda env create -f conda.yaml
  conda activate text-emotion
  python main.py

    then test APIs in postman or Swagger 

3. for image-emotion:
  # For image-emotion
  cd packages/image-emotion
  conda env create -f conda.yaml
  conda activate image-emotion
  python main.py

  then test APIs in postman or Swagger 

4. Use below commands with out creating any virtual environment:
  install packages using: pip install -r requirements.txt
  run command: python main.py 


Creating docker images / Instances:
1. create instances using  docker-compose yaml file like:
  docker-compose build	
2. to start images use below one:
  docker-compose up
**For text-emotion:**
  http://localhost:8001/docs
      or
    http://127.0.0.1:8001/docs
    try this api: http://127.0.0.1:8001/analyze-text  , as other apis not working as expected 

**for image-emotion:**
    http://localhost:8002/docs
        or 
    http://127.0.0.1:8002/docs

