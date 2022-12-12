import wget
saved_model_url_root = "https://marhamilresearch4.blob.core.windows.net/stego-public/saved_models/"
saved_model_name = "cocostuff27_vit_base_5.ckpt"
wget.download(saved_model_url_root + saved_model_name, "cocostuff27_vit_base_5")
