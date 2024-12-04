# Headshot-AI
Headshot augmentation AI, using inpainting Image models

## How It Works

The app is powered by:

- 🚀 [Cerebrium](https://www.cerebrium.ai/) for Serverless GPU 
- 🚀 [Stable Diffuion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) AI model


## Running Locally

To create your own Headshot AI app, follow these steps:

1. Clone the repository:

```
https://github.com/Lakshay-sethi/Headshot-AI.git
```

2. Create a account on cerebrium to make your own API key:

```
https://docs.cerebrium.ai/cerebrium/getting-started/installation
```

3. Fill in anonymised info in the remotetest.py

   - Fill in `url` with your link to the hosted model
   - Fill in `Authorization` with unique Auth Code generated

4. To run the model

 ```
python remotetest.py
```

## Additional Use-Cases

Headshot AI can be easily adapted to support many other use-cases including:

- AI Avatars
  - Anime
  - Portraits
  - Story Illustrations

- Pet Portraits



## License

Is released under the [MIT License](https://choosealicense.com/licenses/mit/).
