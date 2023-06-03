'''
Unified Gate-attention Network in MMKGR (Zheng et. al ICDE 2023)
'''
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn



class UANBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(UANBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        #print(self)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output
    
    def our_attention(self, Q, K, V):
      
        left_att = K*Q          
        right_att = V*Q

        left_liear = nn.Linear(16, 2).cuda()
        
        left_att_lin = left_liear(left_att)

        M = torch.sigmoid(left_att_lin)  #1024,1,2     

        M1 = M[:,:,0:1]    

        M2 = M[:,:,1:2]#1024,1,1
        M1 = M1.repeat(1,1,16)
        M2 = M2.repeat(1,1,16)

        left_att_mask = K*M1       
  
        right_att_mask = Q*M2       
   
        fix_att = torch.mul(left_att_mask, right_att_mask)
      
        QK_score = F.softmax(fix_att, dim=-1)
    
        QK_score_liear = nn.Linear(16, 1).cuda()
        QK_score = QK_score_liear(QK_score)
     
       
        V_att = QK_score.bmm(right_att)
    
        return V_att
      






    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()
        
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)
        V = V.to(torch.float32)
        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.our_attention(Q_, K_, V_)
 
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o)) # (batch_size, max_q_words, input_size)
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
     
        
        V_att = self.multi_head_attention(Q, K, V)
      
        print("V_att",V_att.size()) #1024,1,600
        Q = torch.tensor(Q, dtype = torch.float32).cuda()

        v_ = V*Q
        print("v_ of size",v_.size())
        att_v = v_ * V_att
        print("size of att_v", att_v.size())
        G = torch.sigmoid(att_v)
        V_att = torch.mul(G, att_v)
        out = V_att
        
