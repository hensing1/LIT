FROM deepmi/fastsurfer:gpu-v2.2.0

#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install monai==1.2.0

RUN apt update && apt install -y git && git clone https://github.com/Project-MONAI/GenerativeModels.git && cd GenerativeModels && git reset --hard 7428fce193771e9564f29b91d29e523dd1b6b4cd && python setup.py install

# COPY ./weights/model_axial.pt /inpainting/weights/model_axial.pth
# COPY ./weights/model_coronal.pt /inpainting/weights/model_coronal.pth
# COPY ./weights/model_sagittal.pt /inpainting/weights/model_sagittal.pth

RUN wget https://zenodo.org/records/14497226/files/model_axial.pt -O /inpainting/weights/model_axial.pth
RUN wget https://zenodo.org/records/14497226/files/model_coronal.pt -O /inpainting/weights/model_coronal.pth
RUN wget https://zenodo.org/records/14497226/files/model_sagittal.pt -O /inpainting/weights/model_sagittal.pth

COPY lit /inpainting/lit
COPY run_lit.sh /inpainting/run_lit.sh

RUN mv /fastsurfer/run_fastsurfer.sh /fastsurfer/run_fastsurfer_segmentation.sh


ENV INPAINTING_DIR=/inpainting

RUN echo "cd /inpainting/ && ./run_lit.sh \"\$@"\" >> /fastsurfer/run_fastsurfer.sh
RUN chmod +x /fastsurfer/run_fastsurfer.sh

ENTRYPOINT ["/bin/bash", "/inpainting/run_lit.sh"]


WORKDIR /inpainting
