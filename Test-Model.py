from fastai.vision.all import * #Imports everything from the vision library of fast ai
path = untar_data(URLs.PETS)/'images'
def is_cat(x): return x[0].isupper()
#Tells fastai what kind of dataset we have and how it is structured:
#There are different loaders, in this case we use an image loader
#valid_pct: tells fastai to hold out 20% of data for validation
#the seed allows us to make sure that the same dataset will always give us the same result
#item_tfms: are applied on each item
#batch_tfms: are applied to a batch of items
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))
#tells fastai to create a convolutional neural network (CNN) and specifies what architecture to use
learn = vision_learner(dls, resnet34, metrics=error_rate)


if __name__ == '__main__':
    #defines how many "epochs" (number of times to look at each images) it will go through to train. Another option would be the "fit" function
    learn.fine_tune(1)