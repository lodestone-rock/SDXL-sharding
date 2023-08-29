unet_partition = [
    # in expand
    (r"conv_in.kernel", (None,None,None,'mp',)),
    (r"conv_in.bias", ('mp',)),

    # attention block sharding pattern
    # project resnet to attention
    (r"attentions_\d+.norm.scale", ('mp',)),
    (r"attentions_\d+.norm.bias", ('mp',)),
    (r"attentions_\d+.proj_in.kernel", ('mp', None,)),
    (r"attentions_\d+.proj_in.bias", (None,)),

    # first attention chunk
    # (r"attentions_\d+.transformer_blocks_\d+.norm1.bias", (None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.norm1.scale", (None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn1.to_q.kernel", (None,'mp',)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn1.to_k.kernel", (None,'mp',)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn1.to_v.kernel", (None,'mp',)),

    # (r"attentions_\d+.transformer_blocks_\d+.attn1.to_out_0.kernel", ('mp',None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn1.to_out_0.bias", (None,)),

    # second attention chunk
    # (r"attentions_\d+.transformer_blocks_\d+.norm2.bias", (None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.norm2.scale", (None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn2.to_q.kernel", (None,'mp',)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn2.to_k.kernel", (None,'mp',)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn2.to_v.kernel", (None,'mp',)),

    # (r"attentions_\d+.transformer_blocks_\d+.attn2.to_out_0.kernel", ('mp',None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.attn2.to_out_0.bias", (None,)),

    # (r"attentions_\d+.transformer_blocks_\d+.norm3.bias", (None,)),
    # (r"attentions_\d+.transformer_blocks_\d+.norm3.scale", (None,)),

    # the sharding pattern is the same for both attention so combine it into one regex 
    (r"attentions_\d+.transformer_blocks_\d+.norm\d+.bias", (None,)),
    (r"attentions_\d+.transformer_blocks_\d+.norm\d+.scale", (None,)),
    (r"attentions_\d+.transformer_blocks_\d+.attn\d+.to_q.kernel", (None,'mp',)),
    (r"attentions_\d+.transformer_blocks_\d+.attn\d+.to_k.kernel", (None,'mp',)),
    (r"attentions_\d+.transformer_blocks_\d+.attn\d+.to_v.kernel", (None,'mp',)),

    (r"attentions_\d+.transformer_blocks_\d+.attn\d+.to_out_0.kernel", ('mp',None,)),
    (r"attentions_\d+.transformer_blocks_\d+.attn\d+.to_out_0.bias", (None,)),

    # ff up project and down project
    (r"attentions_\d+.transformer_blocks_\d+.ff.net_0.proj.kernel", (None,'mp',)),
    (r"attentions_\d+.transformer_blocks_\d+.ff.net_0.proj.bias", ('mp',)),
    (r"attentions_\d+.transformer_blocks_\d+.ff.net_2.kernel", ('mp',None,)),
    (r"attentions_\d+.transformer_blocks_\d+.ff.net_2.bias", (None,)),

    # final projection 
    (r"attentions_\d+.proj_out.kernel", (None,'mp',)),
    (r"attentions_\d+.proj_out.bias", ('mp',)),

    # resnet block sharding pattern

    # not sure how to shard this one efficiently 
    # but eh it's only a few layer so don't care
    (r"upsamplers_\d+.conv.bias", ('mp',)),
    (r"upsamplers_\d+.conv.kernel", (None,None,None,'mp',)),
    (r"downsamplers_\d+.conv.bias", ('mp',)),
    (r"downsamplers_\d+.conv.kernel", (None,None,None,'mp',)),

    # some block has no pointwise conv for shortcut hmmm
    # (r"resnets_\d+.norm\d+.scale", ('mp',)),
    # (r"resnets_\d+.norm\d+.bias", ('mp',)),
    # (r"resnets_\d+.conv\d+.kernel", (None,None,None,'mp',)),
    # (r"resnets_\d+.conv\d+.bias", ('mp',)),
    
    (r"resnets_\d+.time_emb_proj.kernel", ('mp',None,)),
    (r"resnets_\d+.time_emb_proj.bias", (None,)),

    # identical resnet sharding pattern
    (r"resnets_\d+.norm\d+.scale", ('mp',)),
    (r"resnets_\d+.norm\d+.bias", ('mp',)),
    (r"resnets_\d+.conv\d+.kernel", (None,None,None,'mp',)),
    (r"resnets_\d+.conv\d+.bias", ('mp',)),

    (r"resnets_\d+.conv_shortcut.kernel", (None,None,None,'mp',)),
    (r"resnets_\d+.conv_shortcut.bias", ('mp',)),

    # time embedding block
    (r"linear_1.kernel", ('mp',None,)),
    (r"linear_1.bias", (None,)),
    (r"linear_2.kernel", (None,'mp',)),
    (r"linear_2.bias", ('mp',)),

    # out project
    (r"conv_norm_out.scale", ('mp',)),
    (r"conv_norm_out.bias", ('mp',)),
    
    # just replicate the last layer, dont bother 
    (r"conv_out.kernel", (None,)),
    (r"conv_out.bias", (None,)),
]

clip_partition = [
    # embbeding layer
    ("text_model.embeddings.token_embedding.embedding", ('mp', None)),
    ("text_model.embeddings.position_embedding.embedding", ('mp', None)),

    # attention blocks
    ("text_model.encoder.layers.\d+.layer_norm1.scale", (None,)),
    ("text_model.encoder.layers.\d+.layer_norm1.bias", (None,)),

    # self attention
    ("text_model.encoder.layers.\d+.self_attn.q_proj.kernel", (None,'mp',)),
    ("text_model.encoder.layers.\d+.self_attn.q_proj.bias", ('mp',)),
    ("text_model.encoder.layers.\d+.self_attn.k_proj.kernel", (None,'mp',)),
    ("text_model.encoder.layers.\d+.self_attn.k_proj.bias", ('mp',)),
    ("text_model.encoder.layers.\d+.self_attn.v_proj.kernel", (None,'mp',)),
    ("text_model.encoder.layers.\d+.self_attn.v_proj.bias", ('mp',)),

    ("text_model.encoder.layers.\d+.self_attn.out_proj.kernel", ('mp',None,)),
    ("text_model.encoder.layers.\d+.self_attn.out_proj.bias", (None,)),

    # MLP head
    ("text_model.encoder.layers.\d+.layer_norm2.scale", (None,)),
    ("text_model.encoder.layers.\d+.layer_norm2.bias", (None,)),
    ("text_model.encoder.layers.\d+.mlp.fc1.kernel", (None,'mp',)),
    ("text_model.encoder.layers.\d+.mlp.fc1.bias", ('mp',)),
    ("text_model.encoder.layers.\d+.mlp.fc2.kernel", ('mp',None,)),
    ("text_model.encoder.layers.\d+.mlp.fc2.bias", (None,)),

    # final output layer normalization
    ("text_model.final_layer_norm.scale", (None,)),
    ("text_model.final_layer_norm.bias", (None,)),

    # add on projection layer for text encoder 2 
    ("text_projection.kernel", (None, 'mp')),
]

# gonna replicate vae instead im tired
vae_partition = []

print()