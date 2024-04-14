Annotate Images Using label studio

### - Sign up

### - Create Project
        - Add name 
        - Import image
        - Select OCR as label setup
            - Clear the labels
            - For labels follow the pattern
                - [key] => eg: Total, subtotal
                - [key]Value => eg: TotalValue, subtotalValue
            - Add
            - Allow zoom and show zoomin and out in control (optional)
            - Adust controls display (optional)
            - Save

### - Export Annotations
        - Select COCO format 
        - Export 

### - Start Application
Note: docker engine and docker-compose is required

````bash
docker-compose up
````
```
    - Got to port mentioned in the docker-compose file
    - Create a folder (project)
    - Add Refrence (Important: use the exported image from label studio and not the original)
    - Add Images 
    - Add annotation (exported result.json file and image id as 0 )
    - Extract Annotated Data (time depends on number of imnage and extracting data length)
```     

