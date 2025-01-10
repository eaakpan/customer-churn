## Running App Locally
1. Build docker image:

   ```
   $ docker build -t churn_app_image .
   ```

2. Run container:
```
    $ docker run --name churn_app_container -p 8000:8000 churn_app_image
```

3. Output will contain:

    INFO:     Uvicorn running on http://0.0.0.0:8000 or (http://127.0.0.1:8000/ | http://localhost:8000)
    - use this url in chrome to see the model frontend
    - use for testing the model


4. Query model
   
    4.1 Via web interface (chrome)
       http://0.0.0.0:8000/docs -> test model

    4.2 Via python client:
       
        client.py
    
    4.3 Via curl request


       $ curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
      For windows:
       $ curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"""features""": [5.1, 3.5, 1.4, 0.2]}'


## For Minikube/Kubernetes Deployment

1. Deploy Docker image to DockerHub
```
   $ docker push eakpan15/churn-dashboard:0.0.1  
```

2. Point to docker within Minikube:
```
   $ & minikube docker-env | Invoke-Expression
``` 
2.2 Load images to confirm correct env:

```
      $ docker images
```

3. Build Docker Image:
```   
   $ docker build -t eakpan15/churn-dashboard:0.0.1 .
```

4. Apply YAML file for deployment
```   
   $ kubectl apply -f kubernetes/deployment.yml
```

5. Expose Minikube deployment:
```   
   $ kubectl expose deployment fast-api --type=LoadBalancer --port=8000
```

6. Start/open service on Minikube:
```   
   $  minikube service fast-api
```

7. Redeploy image/service after making coding change:
```   
   $  kubectl delete -f kubernetes/deployment.yml;docker rmi eakpan15/churn-dashboard:0.0.1;docker build -t eakpan15/churn-dashboard:0.0.1 .;kubectl apply -f kubernetes/deployment.yml
```

## Helpful Commands

- Show Docker Images: ```docker images```
- Start Minikube cluster: ```minikube start```
- Opem Minikube dashboard:```minikube dashboard```
- Show Minikube services: ```minikube service list```
- Point to docker within Minikube(only lasts for terminal session):```& minikube docker-env | Invoke-Expression``` 
- Expose Minkube deployment: ```kubectl expose deployment {name} --type=LoadBalancer --port={port number}```
- Start service on Minikube: ```minikube service {name}```
- Redeploy image/service after making coding change:
```kubectl delete -f {path to deployment yaml};docker rmi {image};docker build -t {image} .;kubectl apply -f {path to yaml}```
- Create cronjob: ```kubectl create -f [path/to/cronjob-name] --save-config```
- Delete cronjob: ```kubectl delete cronjob [cronjob-name]```