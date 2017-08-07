import tensorflow as tf

def convolution(input_, filter_shape, strides = [1,1,1,1], padding = False, activation = None, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)

def deformable_conv2d(x, offset_filter, filter_shape, padding=False, activation = None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_filter - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    o_h, o_w, o_ic, o_oc = offset_filter
    assert f_ic==i_c and o_ic==i_c, "# of input_channel should match but %d, %d, %d"%(i_c, f_ic, o_ic)
    assert o_oc==2*f_h*f_w, "# of output channel in offset_filter should be 2*filter_height*filter_width but %d and %d"%(o_oc, 2*f_h*f_w)

    with tf.variable_scope("deform_conv"):
        offset_map = convolution(x, offset_filter, padding=True, scope="offset_conv") # offset_map : [batch, i_h, i_w, o_oc(=2*f_h*f_w)]
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
    offset_map_h = tf.tile(tf.reshape(offset_map[...,0], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(tf.reshape(offset_map[...,1], [batch, i_h, i_w, f_h, f_w]), [i_c,1,1,1,1]) # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(tf.range(i_w, dtype=tf.float32), tf.range(i_h, dtype=tf.float32)) # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(tf.range(f_w, dtype=tf.float32), tf.range(f_h, dtype=tf.float32)) # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(tf.reshape(coord_h, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_h [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_w = tf.tile(tf.reshape(coord_w, [1, i_h, i_w, 1, 1]), [batch*i_c, 1, 1, f_h, f_w]) # coords_w [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_fh = tf.tile(tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fh [batch*i_c, i_h, i_w, f_h, f_w) 
    coord_fw = tf.tile(tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]), [batch*i_c, i_h, i_w, 1, 1]) # coords_fw [batch*i_c, i_h, i_w, f_h, f_w) 

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(coord_h, clip_value_min = 0, clip_value_max = i_h-1) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(coord_w, clip_value_min = 0, clip_value_max = i_w-1) # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w), tf.int32) # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]), [-1, i_h, i_w]) # [i_c*batch, i_h, i_w]

    bc_index= tf.tile(tf.reshape(tf.range(batch*i_c), [-1,1,1,1,1]), [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hm,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wm,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(values=[tf.expand_dims(bc_index,-1), tf.expand_dims(coord_hM,-1), tf.expand_dims(coord_wM,-1)] , axis=-1) # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm) # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM) # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32) 
    coord_hM = tf.cast(coord_hM, tf.float32) 
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]), [1,2,4,3,5,0]) # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h*f_h, i_w*f_w, i_c]) # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = convolution(x_ip, filter_shape, strides=[1, f_h, f_w, 1], activation=activation, scope="deform_conv")
    return deform_conv
