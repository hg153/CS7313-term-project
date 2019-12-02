"""
File: logger.py
Modified by: Senthil Purushwalkam
Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
Email: spurushw<at>andrew<dot>cmu<dot>edu
Github: https://github.com/senthilps8
Description: 
"""
from matplotlib import pyplot as plt

import tensorflow as tf
from torch.autograd import Variable
import numpy as np
import scipy.misc
import os
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):

    def __init__(self, log_dir, name=None):
        """Create a summary writer logging to log_dir."""
        if name is None:
            name = 'temp'
        self.name = name
        if name is not None:
            try:
                os.makedirs(os.path.join(log_dir, name))
            except:
                pass
            self.writer = tf.summary.create_file_writer(os.path.join(log_dir, name),
                                                filename_suffix=name)
        else:
            self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=name)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        value=value.detach().cpu().numpy()
#        summary = tf.summary.scalar(tag,value,step=step)
        with self.writer.as_default():
            tf.summary.scalar(tag,value,step=step)
#        self.writer.add_summary(summary, step)

    def image_summary4(self, tag, images, step):
        """Log a list of images."""

        """img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)"""
        
        a,b,c,d=images.shape
        images=np.reshape(images.detach().cpu().numpy(),(a,c,d,b))
        
        with self.writer.as_default():
            tf.summary.image(tag,images,step=step)
            
    def image_summary3(self,tag,images,step):
        a,b,c=images.shape
        images=np.reshape(images.detach().cpu().numpy(),(a,b,c,1))

        #save image
        plt.imshow(np.squeeze(images),interpolation='nearest')
        plt.savefig('./result_test/%s_%s.png'%(step,tag.split('/')[1]))
        plt.close
        
        with self.writer.as_default():
            tf.summary.image(tag,images,step=step)
            
    def image_summary2(self,tag,images,step):
        a,b,c=images.shape
        images=np.reshape(images,(a,b,c,1))

        #save image
        plt.imshow(np.squeeze(images),interpolation='nearest')
        plt.savefig('./result_test/%s_%s.png'%(step,tag.split('/')[1]))
        plt.close
        
        with self.writer.as_default():
            tf.summary.image(tag,images,step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        with self.writer.as_default():
            tf.summary.histogram(tag,values,step=step,buckets=bins)

        # Create a histogram using numpy
        """counts, bin_edges = np.histogram(values, bins=bins)
       
        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()"""

    def to_np(self, x):
        return x.data.cpu().numpy()

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def model_param_histo_summary(self, model, step):
        """log histogram summary of model's parameters
        and parameter gradients
        """
        for tag, value in model.named_parameters():
            if value.grad is None:
                continue
            tag = tag.replace('.', '/')
            tag = self.name+'/'+tag
            self.histo_summary(tag, self.to_np(value), step)
            self.histo_summary(tag+'/grad', self.to_np(value.grad), step)

