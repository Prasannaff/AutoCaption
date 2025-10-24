# Image Captioning

Captioning is an img2txt model that uses the BLIP. Exports captions of images.

## Checkpoints [Required]

If there is no 'Checkpoints' folder, the script will automatically create the folder and download the model file, you can do this manually if you want.

Download the fine-tuned checkpoint and copy into 'checkpoints' folder (create if does not exists)

- [BLIP-Large](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth)

## Demo

<img src='./demo.jpg' width=500px>

```txt
Captions--
  |-0_captions.txt
  |-1_captions.txt
  |-2_captions.txt
  |-3_captions.txt
  |-4_captions.txt
```

## Usage

```bash
usage: inference.py [-h] [-i INPUT] [-b BATCH] [-p PATHS] [-g GPU_ID]        

Image caption CLI

optional arguments:
  -h, --help                      show this help message and exit
  -i INPUT,  --input INPUT        Input directoryt path, such as ./images
  -b BATCH,  --batch BATCH        Batch size
  -p PATHS,  --paths PATHS        A any.txt files contains all image paths.
  -g GPU_ID, --gpu-id GPU_ID      gpu device to use (default=0) can be 0,1,2 for multi-gpu
```

### Example

```bash
python inference.py -i /path/images/folder --batch 8 --gpu 0
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.


