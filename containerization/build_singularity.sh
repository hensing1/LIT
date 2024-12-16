set -e


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Build Docker container
docker build . -f ./containerization/Dockerfile -t deepmi/lit:singularity_preparation

# Save Docker container as Singularity image
docker run --privileged -t --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $SCRIPT_DIR/:/output \
    singularityware/docker2singularity \
    deepmi/lit:singularity_preparation /output/deepmi_lit_dev.sif

#Change file ownership to current user
docker run --rm -v $SCRIPT_DIR/:/output --entrypoint bash ubuntu -c "chown $(id -u):$(id -g) /output/deepmi_lit_singularity_preparation*.simg"
mv $SCRIPT_DIR/deepmi_lit_singularity_preparation*.simg $SCRIPT_DIR/deepmi_lit_dev.simg
docker rmi deepmi/lit:singularity_preparation

# ls $HOME/singularity_images/fastsurfer_tumor_0.3-202*.simg
# image=$(ls $HOME/singularity_images/fastsurfer_tumor_0.3-202*.simg)
# # change file ownership
# docker run --rm -v $HOME/singularity_images/:/output ubuntu chown $(id -u):$(id -g) /output/$image
# mv $image $HOME/singularity_images/fastsurfer_tumor_0.3.sif


# TODO: test this
# Save Docker container as Singularity image using singularity pull
#singularity pull docker://deepmi/lit:singularity_preparation