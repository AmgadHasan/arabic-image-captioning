ńÁ
äł
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
Ž
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeíout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ű

cnn_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecnn_encoder/dense/bias
~
*cnn_encoder/dense/bias/Read/ReadVariableOpReadVariableOpcnn_encoder/dense/bias*
_output_shapes	
:*
dtype0

cnn_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namecnn_encoder/dense/kernel

,cnn_encoder/dense/kernel/Read/ReadVariableOpReadVariableOpcnn_encoder/dense/kernel* 
_output_shapes
:
*
dtype0

serving_default_input_1Placeholder*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype0*!
shape:˙˙˙˙˙˙˙˙˙@
ń
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn_encoder/dense/kernelcnn_encoder/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1776771

NoOpNoOp
ĺ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0* 
valueB B
Ĺ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc
	
signatures*


0
1*


0
1*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses


kernel
bias*

serving_default* 
XR
VARIABLE_VALUEcnn_encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcnn_encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
* 

0*
* 
* 
* 
* 
* 
* 
* 


0
1*


0
1*
* 

non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

!trace_0* 

"trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ó
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecnn_encoder/dense/kernelcnn_encoder/dense/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1776886
Î
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn_encoder/dense/kernelcnn_encoder/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1776902á
´
É
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776722
input_1!
dense_1776716:

dense_1776718:	
identity˘dense/StatefulPartitionedCallđ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1776716dense_1776718*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1776715z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_1
­
ę
#__inference__traced_restore_1776902
file_prefix=
)assignvariableop_cnn_encoder_dense_kernel:
8
)assignvariableop_1_cnn_encoder_dense_bias:	

identity_3˘AssignVariableOp˘AssignVariableOp_1Ű
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ź
AssignVariableOpAssignVariableOp)assignvariableop_cnn_encoder_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ŕ
AssignVariableOp_1AssignVariableOp)assignvariableop_1_cnn_encoder_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: p
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Ô
 __inference__traced_save_1776886
file_prefixC
/read_disablecopyonread_cnn_encoder_dense_kernel:
>
/read_1_disablecopyonread_cnn_encoder_dense_bias:	
savev2_const

identity_5˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead/read_disablecopyonread_cnn_encoder_dense_kernel"/device:CPU:0*
_output_shapes
 ­
Read/ReadVariableOpReadVariableOp/read_disablecopyonread_cnn_encoder_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_1/DisableCopyOnReadDisableCopyOnRead/read_1_disablecopyonread_cnn_encoder_dense_bias"/device:CPU:0*
_output_shapes
 Ź
Read_1/ReadVariableOpReadVariableOp/read_1_disablecopyonread_cnn_encoder_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ř
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: ˝
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_5Identity_5:output:0*
_input_shapes

: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
š

%__inference_signature_wrapper_1776771
input_1
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallŔ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1776680t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_1
ž
ü
B__inference_dense_layer_call_and_return_conditional_losses_1776715

inputs5
!tensordot_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::íĎY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ř

'__inference_dense_layer_call_fn_1776820

inputs
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1776715t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ç

-__inference_cnn_encoder_layer_call_fn_1776741
input_1
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallć
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776734t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_1
ž
ü
B__inference_dense_layer_call_and_return_conditional_losses_1776851

inputs5
!tensordot_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::íĎY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
÷ 
˘
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776811
image_features;
'dense_tensordot_readvariableop_resource:
4
%dense_biasadd_readvariableop_resource:	
identity˘dense/BiasAdd/ReadVariableOp˘dense/Tensordot/ReadVariableOp
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype0^
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:e
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       a
dense/Tensordot/ShapeShapeimage_features*
T0*
_output_shapes
::íĎ_
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ×
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:_
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: a
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: ]
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense/Tensordot/transpose	Transposeimage_featuresdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:_
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@a

dense/ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@l
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:\ X
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameimage_features
É
Đ
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776734
image_features!
dense_1776728:

dense_1776730:	
identity˘dense/StatefulPartitionedCall÷
dense/StatefulPartitionedCallStatefulPartitionedCallimage_featuresdense_1776728dense_1776730*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_1776715z
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@f
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:\ X
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameimage_features
ü
Ľ
-__inference_cnn_encoder_layer_call_fn_1776780
image_features
unknown:

	unknown_0:	
identity˘StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallimage_featuresunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:˙˙˙˙˙˙˙˙˙@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776734t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(
_user_specified_nameimage_features
Î&
Ľ
"__inference__wrapped_model_1776680
input_1G
3cnn_encoder_dense_tensordot_readvariableop_resource:
@
1cnn_encoder_dense_biasadd_readvariableop_resource:	
identity˘(cnn_encoder/dense/BiasAdd/ReadVariableOp˘*cnn_encoder/dense/Tensordot/ReadVariableOp 
*cnn_encoder/dense/Tensordot/ReadVariableOpReadVariableOp3cnn_encoder_dense_tensordot_readvariableop_resource* 
_output_shapes
:
*
dtype0j
 cnn_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 cnn_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       f
!cnn_encoder/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
::íĎk
)cnn_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
$cnn_encoder/dense/Tensordot/GatherV2GatherV2*cnn_encoder/dense/Tensordot/Shape:output:0)cnn_encoder/dense/Tensordot/free:output:02cnn_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+cnn_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&cnn_encoder/dense/Tensordot/GatherV2_1GatherV2*cnn_encoder/dense/Tensordot/Shape:output:0)cnn_encoder/dense/Tensordot/axes:output:04cnn_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!cnn_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ¤
 cnn_encoder/dense/Tensordot/ProdProd-cnn_encoder/dense/Tensordot/GatherV2:output:0*cnn_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#cnn_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ş
"cnn_encoder/dense/Tensordot/Prod_1Prod/cnn_encoder/dense/Tensordot/GatherV2_1:output:0,cnn_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'cnn_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ä
"cnn_encoder/dense/Tensordot/concatConcatV2)cnn_encoder/dense/Tensordot/free:output:0)cnn_encoder/dense/Tensordot/axes:output:00cnn_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ż
!cnn_encoder/dense/Tensordot/stackPack)cnn_encoder/dense/Tensordot/Prod:output:0+cnn_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
%cnn_encoder/dense/Tensordot/transpose	Transposeinput_1+cnn_encoder/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@Ŕ
#cnn_encoder/dense/Tensordot/ReshapeReshape)cnn_encoder/dense/Tensordot/transpose:y:0*cnn_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Á
"cnn_encoder/dense/Tensordot/MatMulMatMul,cnn_encoder/dense/Tensordot/Reshape:output:02cnn_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#cnn_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:k
)cnn_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ď
$cnn_encoder/dense/Tensordot/concat_1ConcatV2-cnn_encoder/dense/Tensordot/GatherV2:output:0,cnn_encoder/dense/Tensordot/Const_2:output:02cnn_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ş
cnn_encoder/dense/TensordotReshape,cnn_encoder/dense/Tensordot/MatMul:product:0-cnn_encoder/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
(cnn_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp1cnn_encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ł
cnn_encoder/dense/BiasAddBiasAdd$cnn_encoder/dense/Tensordot:output:00cnn_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@y
cnn_encoder/dense/ReluRelu"cnn_encoder/dense/BiasAdd:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@x
IdentityIdentity$cnn_encoder/dense/Relu:activations:0^NoOp*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
NoOpNoOp)^cnn_encoder/dense/BiasAdd/ReadVariableOp+^cnn_encoder/dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@: : 2T
(cnn_encoder/dense/BiasAdd/ReadVariableOp(cnn_encoder/dense/BiasAdd/ReadVariableOp2X
*cnn_encoder/dense/Tensordot/ReadVariableOp*cnn_encoder/dense/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_1"ó
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ľ
serving_defaultĄ
@
input_15
serving_default_input_1:0˙˙˙˙˙˙˙˙˙@A
output_15
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙@tensorflow/serving/predict:0
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc
	
signatures"
_tf_keras_model
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¸
trace_0
trace_12
-__inference_cnn_encoder_layer_call_fn_1776741
-__inference_cnn_encoder_layer_call_fn_1776780 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
î
trace_0
trace_12ˇ
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776722
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776811 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
ÍBĘ
"__inference__wrapped_model_1776680input_1"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses


kernel
bias"
_tf_keras_layer
,
serving_default"
signature_map
,:*
2cnn_encoder/dense/kernel
%:#2cnn_encoder/dense/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŕBÝ
-__inference_cnn_encoder_layer_call_fn_1776741input_1" 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
çBä
-__inference_cnn_encoder_layer_call_fn_1776780image_features" 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
űBř
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776722input_1" 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B˙
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776811image_features" 
˛
FullArgSpec
args
jimage_features
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
non_trainable_variables

layers
metrics
layer_regularization_losses
 layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
á
!trace_02Ä
'__inference_dense_layer_call_fn_1776820
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z!trace_0
ü
"trace_02ß
B__inference_dense_layer_call_and_return_conditional_losses_1776851
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z"trace_0
ĚBÉ
%__inference_signature_wrapper_1776771input_1"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŃBÎ
'__inference_dense_layer_call_fn_1776820inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ěBé
B__inference_dense_layer_call_and_return_conditional_losses_1776851inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
"__inference__wrapped_model_1776680u
5˘2
+˘(
&#
input_1˙˙˙˙˙˙˙˙˙@
Ş "8Ş5
3
output_1'$
output_1˙˙˙˙˙˙˙˙˙@ş
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776722n
5˘2
+˘(
&#
input_1˙˙˙˙˙˙˙˙˙@
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙@
 Á
H__inference_cnn_encoder_layer_call_and_return_conditional_losses_1776811u
<˘9
2˘/
-*
image_features˙˙˙˙˙˙˙˙˙@
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙@
 
-__inference_cnn_encoder_layer_call_fn_1776741c
5˘2
+˘(
&#
input_1˙˙˙˙˙˙˙˙˙@
Ş "&#
unknown˙˙˙˙˙˙˙˙˙@
-__inference_cnn_encoder_layer_call_fn_1776780j
<˘9
2˘/
-*
image_features˙˙˙˙˙˙˙˙˙@
Ş "&#
unknown˙˙˙˙˙˙˙˙˙@ł
B__inference_dense_layer_call_and_return_conditional_losses_1776851m
4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙@
Ş "1˘.
'$
tensor_0˙˙˙˙˙˙˙˙˙@
 
'__inference_dense_layer_call_fn_1776820b
4˘1
*˘'
%"
inputs˙˙˙˙˙˙˙˙˙@
Ş "&#
unknown˙˙˙˙˙˙˙˙˙@Ş
%__inference_signature_wrapper_1776771
@˘=
˘ 
6Ş3
1
input_1&#
input_1˙˙˙˙˙˙˙˙˙@"8Ş5
3
output_1'$
output_1˙˙˙˙˙˙˙˙˙@