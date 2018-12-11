#coding:utf-8
import tensorflow as tf

#L1 loss 只是对正样本进行操作
def build_smooth_L1_loss(predict_loc_tensor, ground_loc_tensor, weights):
    """
    :param predict_loc_tensor:  预测的信息
    :param ground_loc_tensor:   编码后的ground信息
    :param weights:             权重信息,包含有效和如果有效占用的比例信息
    :return:
    """
    #按照 paper 中实现的 L1 loss
    abs_diff = tf.abs(predict_loc_tensor - ground_loc_tensor)

    L1_loss = tf.where(tf.less(abs_diff, 1), 0.5 * tf.pow(abs_diff, 2), abs_diff - 0.5)

    L1_loss = L1_loss * weights

    loc_loss = tf.reduce_mean(tf.reduce_sum(L1_loss, axis=-1), name='location_loss')

    return loc_loss


#log loss weights 正负样本权重均在里面
def build_log_loss(predict_loc_tensor, ground_loc_tensor, weights):

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=ground_loc_tensor, logits = predict_loc_tensor)

    weighted_cross_loss = tf.reduce_sum(loss * weights)

    weights_div = tf.reduce_sum(weights)

    def f1():

        return 0.0

    def f2():

        return weighted_cross_loss/weights_div

    weighted_cross_loss = tf.cond(tf.equal(weights_div, 0.0), f1, f2)

    return weighted_cross_loss


def build_log_loss_adv(predict_loc_tensor, ground_loc_tensor, weights):

    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=ground_loc_tensor, logits = predict_loc_tensor)

    loss = tf.reduce_sum(-1.0 * tf.log(tf.nn.softmax(predict_loc_tensor)) * ground_loc_tensor, axis=1)

    weighted_cross_loss = tf.reduce_sum(loss * weights)

    weights_div = tf.reduce_sum(weights)

    def f1():

        return 0.0

    def f2():

        return weighted_cross_loss/weights_div

    weighted_cross_loss = tf.cond(tf.equal(weights_div, 0.0), f1, f2)

    return weighted_cross_loss



def build_localization_loss(logistic, ground_truth, weights):

    loss = build_smooth_L1_loss(logistic, ground_truth, weights)

    return loss



def build_classification_loss(logistic, ground_truth, weights):

    loss = build_log_loss(logistic, ground_truth, weights)
    # loss = build_log_loss_adv(logistic, ground_truth, weights)

    return loss


#tf.while_loop的做法
def build_ssd_loss(logistic_tensor, ground_truth_box_label_scores_tensor, total_anchor_number, neg_pos_ratio = 3.0, min_negative_number = 0, alpha = 1.0):

    total_localization_loss   = tf.zeros(shape= [1], dtype=tf.float32)
    total_classification_loss = tf.zeros(shape= [1],dtype=tf.float32)

    batch_size = tf.shape(ground_truth_box_label_scores_tensor)[0]

    def condition(i, total_localization_loss, total_classification_loss, logistic_tensor, ground_truth_box_label_scores_tensor):

        r = tf.less(i, tf.shape(ground_truth_box_label_scores_tensor)[0])

        return r


    def body(i, total_localization_loss, total_classification_loss, logistic_tensor, ground_truth_box_label_scores_tensor):

        ####  300 * 300 的输入来说 current_logistic_tensor = 8732 * (4  + 21)
        current_logistic_tensor = logistic_tensor[i]

        ####  current_ground_truth_box_label_scores_tensor 8732 * ( 21 (one hot labe) + 4(anchor encode box) + 1 (score) + 4(gt box) )
        current_ground_truth_box_label_scores_tensor = ground_truth_box_label_scores_tensor[i]

        current_gt_label       = current_ground_truth_box_label_scores_tensor[:,0:21]
        current_gt_encode_box  = current_ground_truth_box_label_scores_tensor[:,21:25]
        current_gt_score       = current_ground_truth_box_label_scores_tensor[:,25]

        #预测的负样本置信度
        logistic_negative_confidence      = tf.nn.softmax(current_logistic_tensor[:,0:21])[:,0]

        #查找负样本的置信度 按低->高 排序, 但是利用 tf.nn.top_k 这个函数是做高->低 (前K个) 因此取负号
        reversed_logistic_negative_confidence = logistic_negative_confidence * (-1.0)


        logistic_encode_box = current_logistic_tensor[:, 21:25]
        logistic_encode_label = current_logistic_tensor[:, 0:21]


        #计算正样本个数
        current_all_positive_index = tf.greater(current_gt_score, 0.0) #current_gt_score > 0.0  正样本index
        current_all_negative_index = tf.equal(current_gt_score, 0.0)   #current_gt_score == 0.0  负样本index


        current_all_postive_number = tf.count_nonzero(tf.cast(current_all_positive_index, tf.int32), dtype=tf.int32) #正样本个数

        #计算负样本最大个数
        #当前得到的所有负样本个数
        current_all_negative_number = total_anchor_number - tf.cast(current_all_postive_number,dtype=tf.int32)


        #当前可得到的最多的负样本个数
        current_valid_negative_number =  tf.cast(tf.cast(current_all_postive_number,dtype=tf.float32) * neg_pos_ratio, dtype=tf.int32)
        current_valid_negative_number =  tf.minimum(current_all_negative_number, current_valid_negative_number)
        current_valid_negative_number =  tf.maximum(current_valid_negative_number, min_negative_number)

        #
        reversed_logistic_negative_confidence = tf.where(current_all_negative_index, reversed_logistic_negative_confidence, (-1.0) * tf.ones_like(reversed_logistic_negative_confidence))

        valid_negative_value, valid_negative_indices = tf.nn.top_k(reversed_logistic_negative_confidence, current_valid_negative_number, sorted=False)

        #在 logistic 中取得有效的(经过排序得到)负样本 的index
        current_valid_logistic_negative_index_weights = tf.sparse_to_dense(valid_negative_indices, [total_anchor_number], 1.0, 0.0, validate_indices = False)

        current_valid_logistic_positive_index_weights = tf.cast(current_all_positive_index,dtype=tf.float32) #所有的正样本均为有效的


        #计算 localization loss
        #在BP的时候去除  current_valid_logistic_positive_index_weights 的干扰 不进行反向传播
        current_valid_logistic_positive_index_weights_stop = tf.stop_gradient(current_valid_logistic_positive_index_weights)
        loc_loss = build_localization_loss(logistic_encode_box, current_gt_encode_box, current_valid_logistic_positive_index_weights_stop)

        #计算 classification loss
        current_valid_classification_index_weights =  tf.cast(tf.logical_or(current_valid_logistic_negative_index_weights > 0.0, current_valid_logistic_positive_index_weights > 0.0), dtype=tf.float32)
        current_valid_classification_index_weights_stop = tf.stop_gradient(current_valid_classification_index_weights)
        cls_loss = build_classification_loss(logistic_encode_label, current_gt_label, current_valid_classification_index_weights_stop)

        total_localization_loss += loc_loss
        total_classification_loss += cls_loss

        i = i + 1

        return i, total_localization_loss, total_classification_loss, logistic_tensor, ground_truth_box_label_scores_tensor


    i = 0

    i, total_localization_loss, total_classification_loss, logistic_tensor, ground_truth_box_label_scores_tensor = \
    tf.while_loop(cond=condition, body=body,loop_vars=[i, total_localization_loss, total_classification_loss, logistic_tensor, ground_truth_box_label_scores_tensor])



    return total_localization_loss / tf.cast(batch_size, dtype=tf.float32), total_classification_loss / tf.cast(batch_size, dtype=tf.float32)


#使用展开的方式进行
def build_ssd_loss_adv(logistic_tensor, ground_truth_box_label_scores_tensor, total_anchor_number, neg_pos_ratio = 3.0, min_negative_number = 0, alpha = 1.0, total_class = 21):

    logistic_shape = tf.shape(logistic_tensor)
    batch_size = tf.cast(logistic_shape[0], tf.uint8)

    predict_label_tensor         = logistic_tensor[:, :, 0:21]
    predict_encoded_bbox_tensor  = logistic_tensor[:, :, 21:25]


    ground_label_tensor = ground_truth_box_label_scores_tensor[:,:,0:21] #ground_label_tensor 格式为 [batch_size, anchors, 21]


    #ground_label_sparse_tensor 大小为 [batch_size, anchors] 记录的是每个anchor 的label标记 范围 : 0 - 20
    ground_label_sparse_tensor = tf.reshape(tf.argmax(ground_label_tensor, axis=-1, output_type=tf.int32), [-1, total_anchor_number]) #ground_label_sparse_tensor [batce size total_anchor_number]

    ground_bbox_tensor  = ground_truth_box_label_scores_tensor[:,:, 21:25]         #gt的box
    ground_bbox_match_scores_tensor = ground_truth_box_label_scores_tensor[:,:,25] #匹配scores

    #转换成平整的 [batch_size * 8732(300的图像), 21(4)]
    ground_label_flatten_tensor             = tf.reshape(ground_label_tensor, [-1, 21])   #展平操作 [batch_size * all_anchor, 21]
    ground_bbox_flatten_tensor              = tf.reshape(ground_bbox_tensor , [-1, 4])    #展平操作 [batch_size * all_anchor, 4]
    ground_bbox_match_scores_flatten_tensor = tf.reshape(ground_bbox_match_scores_tensor , [-1]) #展平操作 [batch_size * all_anchor]

    pre_label_flatten_tensor = tf.reshape(logistic_tensor[:, :, 0: 21], shape=[-1, 21])  #展平操作 [batch_size * all_anchor, 21]
    pre_bbox_flatten_tensor  = tf.reshape(logistic_tensor[:, :, 21:25], shape=[-1, 4])   #展平操作 [batch_size * all_anchor, 4]


    #计算求得正负样本的index(根据比例来进行)
    batch_max_pos_sample_mask = tf.reshape(tf.greater(ground_label_sparse_tensor ,0 ),[-1])  #label不为0的作为正样本 True 或者 False 作为mask shape [batch_size * anchor]
    batch_max_neg_sample_mask = tf.equal(ground_label_sparse_tensor, 0) #label 等于 0的 anchor 作为负样本 但是这个样本个数是最大的负样本 比实际可以选择的负样本要多很多 下面会进行筛选


    #得到了每个image中最多有多少个正样本和负样本的个数 batch_size大小的
    batch_pos_sample_max_number = tf.count_nonzero(ground_label_sparse_tensor, axis=-1, dtype=tf.int32)  #shape [batch size, 1]
    batch_neg_sample_max_number = tf.count_nonzero(tf.cast(batch_max_neg_sample_mask, tf.uint8), axis=-1, dtype=tf.int32) #shape [batch size, 1]

    batch_neg_sample_valid_number  = batch_pos_sample_max_number * tf.cast(neg_pos_ratio,tf.int32) #每张图片 可以选择的按照比例的负样本个数 但是这个个数不一定是最终的
    batch_neg_sample_select_number = tf.minimum(batch_neg_sample_max_number, batch_neg_sample_valid_number) #这个是batch中 每个Image可以选择的负样本个数
    batch_neg_sample_select_number = tf.maximum(batch_neg_sample_select_number, min_negative_number)  # 这个是batch中 每个Image可以选择的负样本个数 shape [batch size, 1]

    #求batch中每一个image的负样本位置信息(对应在anchor上)
    predict_for_bg = tf.nn.softmax(predict_label_tensor,axis=-1)[:, :, 0]  # predict_for_bg shape 为 [batch_size , all_anchor, 1] 负样本的预测概率值
    predict_for_bg = tf.reshape(predict_for_bg, [-1, total_anchor_number]) # predict_for_bg shape [batch_size , total_anchor_number]

    prob_for_neg   = tf.where(batch_max_neg_sample_mask, 0. - predict_for_bg , 0. - tf.ones_like(predict_for_bg)) #prob_for_neg shape [batch size , total_anchor_number]

    topk_prob_for_bg, _ = tf.nn.top_k(prob_for_neg, k=tf.shape(prob_for_neg)[1]) # topk_prob_for_bg shape 是 [batch size , total_anchor_number]

    #负样本的pro大->小排序 低k个坐标下的值
    score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(logistic_shape[0]), batch_neg_sample_select_number - 1], axis=-1)) #score_at_k [ batch_size ] 个数字

    selected_valid_neg_mask = prob_for_neg >= tf.expand_dims(score_at_k, axis=-1) #selected_valid_neg_mask shape [batch_size, total_anchor_number]

    #得到了训练用的mask 包含 正样本和负样本
    final_mask = tf.stop_gradient(tf.logical_or(tf.reshape(tf.logical_and(batch_max_neg_sample_mask, selected_valid_neg_mask), [-1]), batch_max_pos_sample_mask))

    total_sample = tf.count_nonzero(final_mask)

    pre_valid_label = tf.boolean_mask(pre_label_flatten_tensor, final_mask)
    pre_valid_bbox  = tf.boolean_mask(pre_bbox_flatten_tensor , tf.stop_gradient(batch_max_pos_sample_mask))

    ground_valid_label = tf.boolean_mask(ground_label_flatten_tensor, final_mask)
    ground_valid_bbox  = tf.stop_gradient(tf.boolean_mask(ground_bbox_flatten_tensor , batch_max_pos_sample_mask))

    #交叉熵损失
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ground_valid_label, logits = pre_valid_label))

    loc_loss = build_smooth_L1_loss(pre_valid_bbox, ground_valid_bbox, 1.0)

    #测试使用
    tf.add_to_collection('neg_number', batch_neg_sample_valid_number)
    tf.add_to_collection('pos_number', batch_pos_sample_max_number)

    return loc_loss, cross_entropy
