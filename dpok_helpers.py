import numpy as np
import torch
import torch.nn.functional as F
import copy
import dataclasses
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_random_indices(num_indices, sample_size):
	"""Returns a random sample of indices from a larger list of indices.

	Args:
			num_indices (int): The total number of indices to choose from.
			sample_size (int): The number of indices to choose.

	Returns:
			A numpy array of `sample_size` randomly chosen indices.
	"""
	return np.random.choice(num_indices, size=sample_size, replace=False)


def get_test_prompts(flag):
	"""Gets test prompts."""

	if flag == "drawbench":
		test_batch = [
				"A pink colored giraffe.",
				(
						"An emoji of a baby panda wearing a red hat, green gloves, red"
						" shirt, and green pants."
				),
				"A blue bird and a brown bear.",
				"A yellow book and a red vase.",
				"Three dogs on the street.",
				"Two cats and one dog sitting on the grass.",
				"A wine glass on top of a dog.",
				"A cube made of denim. A cube with the texture of denim.",
		]
	elif flag == "partiprompt":
		test_batch = [
				"a panda bear with aviator glasses on its head",
				"Times Square during the day",
				"the skyline of New York City",
				"square red apples on a tree with circular green leaves",
				"a map of Italy",
				"a sketch of a horse",
				"the word 'START' on a blue t-shirt",
				"a dolphin in an astronaut suit on saturn",
		]
	elif flag == "coco":
		test_batch = [
				"A Christmas tree with lights and teddy bear",
				"A group of planes near a large wall of windows.",
				"three men riding horses through a grassy field",
				"A man and a woman posing in front of a motorcycle.",
				"A man sitting on a motorcycle smoking a cigarette.",
				"A pear, orange, and two bananas in a wooden bowl.",
				"Some people posting in front of a camera for a picture.",
				"Some very big furry brown bears in a big grass field.",
		]
	elif flag == "paintskill":
		test_batch = [
				"a photo of blue bear",
				"a photo of blue fire hydrant",
				"a photo of bike and skateboard; skateboard is left to bike",
				"a photo of bed and human; human is right to bed",
				"a photo of suitcase and bench; bench is left to suitcase",
				"a photo of bed and stop sign; stop sign is above bed",
				(
						"a photo of dining table and traffic light; traffic light is below"
						" dining table"
				),
				"a photo of bear and bus; bus is above bear",
		]
	else:
		test_batch = [
				"A dog and a cat.",
				"A cat and a dog.",
				"Two dogs in the park.",
				"Three dogs in the park.",
				"Four dogs in the park.",
				"A blue colored rabbit.",
				"A red colored rabbit.",
				"A green colored rabbit.",
		]

	return test_batch


def _update_output_dir(args):
	"""Modifies `args.output_dir` using configurations in `args`.

	Args:
			args: argparse.Namespace object.
	"""
	if args.single_flag == 1:
		data_log = "single_prompt/" + args.single_prompt.replace(" ", "_") + "/"
	else:
		data_log = args.prompt_path.split("/")[-2] + "_"
		data_log += args.prompt_category + "/"
	learning_log = "p_lr" + str(args.learning_rate) + "_s" + str(args.p_step)
	learning_log += (
			"_b"
			+ str(args.p_batch_size)
			+ "_g"
			+ str(args.gradient_accumulation_steps)
	)
	learning_log += "_l" + str(args.lora_rank)
	coeff_log = "_kl" + str(args.kl_weight) + "_re" + str(args.reward_weight)
	if args.kl_warmup > 0:
		coeff_log += "_klw" + str(args.kl_warmup)
	if args.sft_initialization == 0:
		start_log = "/pre_train/"
	else:
		start_log = "/sft/"
	if args.reward_flag == 0:
		args.output_dir += "/img_reward_{}/".format(args.reward_filter)
	else:
  # The above code is a comment in Python. Comments are used to provide explanations or notes within
  # the code and are not executed by the Python interpreter. In this case, the comment is using the
  # "#" symbol to indicate that the following text is a comment and not actual code.
		args.output_dir += "/prev_reward_{}/".format(args.reward_filter)
	args.output_dir += start_log + data_log + "/" + learning_log + coeff_log
	if args.v_flag == 1:
		value_log = "_v_lr" + str(args.v_lr) + "_b" + str(args.v_batch_size)
		value_log += "_s" + str(args.v_step)
		args.output_dir += value_log


'''def _calculate_reward_ir(
		pipe,
		args,
		reward_tokenizer,
		tokenizer,
		weight_dtype,
		reward_clip_model,
		image_reward,
		imgs,
		prompts,
		test_flag=False,
):
	"""Computes reward using ImageReward model."""
	if test_flag:
		image_pil = imgs
	else:
		image_pil = pipe.numpy_to_pil(imgs)[0]
	blip_reward, _ = utils.image_reward_get_reward(
			image_reward, image_pil, prompts, weight_dtype
	)
	if args.reward_filter == 1:
		blip_reward = torch.clamp(blip_reward, min=0)
	inputs = reward_tokenizer(
			prompts,
			max_length=tokenizer.model_max_length,
			padding="do_not_pad",
			truncation=True,
	)
	input_ids = inputs.input_ids
	padded_tokens = reward_tokenizer.pad(
			{"input_ids": input_ids}, padding=True, return_tensors="pt"
	)
	txt_emb = reward_clip_model.get_text_features(
			input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0)
	)
	return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0)'''


def _calculate_reward_custom(
		pipe,
		_,
		reward_tokenizer,
		tokenizer,
		weight_dtype,
		reward_clip_model,
		reward_processor,
		reward_model,
		imgs,
		prompts,
		test_flag=False,
):
	"""Computes reward using custom reward model."""
	# img
	if test_flag:
		image_pil = imgs
	else:
		image_pil = pipe.numpy_to_pil(imgs)[0]
	pixels = (
			reward_processor(images=image_pil.convert("RGB"), return_tensors="pt")
			.pixel_values.to(weight_dtype)
			.to(device)
	)
	img_emb = reward_clip_model.get_image_features(pixels)
	# prompt
	inputs = reward_tokenizer(
			prompts,
			max_length=tokenizer.model_max_length,
			padding="do_not_pad",
			truncation=True,
	)
	input_ids = inputs.input_ids
	padded_tokens = reward_tokenizer.pad(
			{"input_ids": input_ids}, padding=True, return_tensors="pt"
	)
	txt_emb = reward_clip_model.get_text_features(
			input_ids=padded_tokens.input_ids.to(device).unsqueeze(0)
	)
	score = reward_model(txt_emb, img_emb)
	return score.to(weight_dtype).squeeze(0).squeeze(0), txt_emb.squeeze(0)


def _get_batch(data_iter_loader, data_iterator, prompt_list,g_batch_size, num_samples , accelerator):
	"""Creates a batch."""
	batch = next(data_iter_loader, None)
	if batch is None:
		batch = next(
				iter(
						accelerator.prepare(
								data_iterator(prompt_list, batch_size=g_batch_size)
						)
				)
		)

	batch_list = []
	for i in range(len(batch)):
		batch_list.extend([batch[i] for _ in range(num_samples)])
	batch = batch_list
	return batch


def _trim_buffer(buffer_size, state_dict):
	"""Delete old samples from the bufffer."""
	if state_dict["state"].shape[0] > buffer_size:
		state_dict["prompt"] = state_dict["prompt"][-buffer_size:]
		state_dict["state"] = state_dict["state"][-buffer_size:]
		state_dict["next_state"] = state_dict["next_state"][-buffer_size:]
		state_dict["timestep"] = state_dict["timestep"][-buffer_size:]
		state_dict["final_reward"] = state_dict["final_reward"][-buffer_size:]
		state_dict["unconditional_prompt_embeds"] = state_dict[
				"unconditional_prompt_embeds"
		][-buffer_size:]
		state_dict["guided_prompt_embeds"] = state_dict["guided_prompt_embeds"][
				-buffer_size:
		]
		state_dict["txt_emb"] = state_dict["txt_emb"][-buffer_size:]
		state_dict["log_prob"] = state_dict["log_prob"][-buffer_size:]


def _save_model(args, count, is_ddp, accelerator, unet):
	"""Saves UNET model."""
	save_path = os.path.join(args.output_dir, f"save_{count}")
	print(f"Saving model to {save_path}")
	if is_ddp:
		unet_to_save = copy.deepcopy(accelerator.unwrap_model(unet)).to(
				torch.float32
		)
		unet_to_save.save_attn_procs(save_path)
	else:
		unet_to_save = copy.deepcopy(unet).to(torch.float32)
		unet_to_save.save_attn_procs(save_path)


def _collect_rollout(g_step, pipe, is_ddp, batch, calculate_reward, state_dict,step):
	"""Collects trajectories."""
	for _ in range(g_step):
		# samples for each prompt
		# collect the rollout data from the custom sampling function
		# (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
		with torch.no_grad():
			(
					image,
					latents_list,
					unconditional_prompt_embeds,
					guided_prompt_embeds,
					log_prob_list,
					_,
			) = pipe.forward_collect_traj_ddim(prompt=batch, is_ddp=is_ddp,output_type="pil")
			reward_list = []
			txt_emb_list = []
			for i in range(len(batch)):
				reward, txt_emb = calculate_reward([image[i]], batch[i],step)
				reward_list.append(reward)
				txt_emb_list.append(txt_emb)
			reward_list = torch.stack(reward_list).detach().cpu()
			txt_emb_list = torch.stack(txt_emb_list).detach().cpu()
			# store the rollout data
			for i in range(len(latents_list) - 1):
				# deal with a batch of data in each step i
				state_dict["prompt"].extend(batch)
				state_dict["state"] = torch.cat((state_dict["state"], latents_list[i]))
				state_dict["next_state"] = torch.cat(
						(state_dict["next_state"], latents_list[i + 1])
				)
				state_dict["timestep"] = torch.cat(
						(state_dict["timestep"], torch.LongTensor([i] * len(batch)))
				)
				state_dict["final_reward"] = torch.cat(
						(state_dict["final_reward"], reward_list)
				)
				state_dict["unconditional_prompt_embeds"] = torch.cat((
						state_dict["unconditional_prompt_embeds"],
						unconditional_prompt_embeds,
				))
				state_dict["guided_prompt_embeds"] = torch.cat(
						(state_dict["guided_prompt_embeds"], guided_prompt_embeds)
				)
				state_dict["txt_emb"] = torch.cat((state_dict["txt_emb"], txt_emb_list))
				state_dict["log_prob"] = torch.cat(
						(state_dict["log_prob"], log_prob_list[i])
				)
			del (
					image,
					latents_list,
					unconditional_prompt_embeds,
					guided_prompt_embeds,
					reward_list,
					txt_emb_list,
					log_prob_list,
					reward,
					txt_emb,
			)
			torch.cuda.empty_cache()


def _train_value_func(value_function, state_dict, accelerator, v_batch_size,v_step):
	"""Trains the value function."""
	indices = get_random_indices(state_dict["state"].shape[0], v_batch_size)
	# permutation = torch.randperm(state_dict['state'].shape[0])
	# indices = permutation[:v_batch_size]
	batch_state = state_dict["state"][indices]
	batch_timestep = state_dict["timestep"][indices]
	batch_final_reward = state_dict["final_reward"][indices]
	batch_txt_emb = state_dict["txt_emb"][indices]
	pred_value = value_function(
			batch_state.to(device).detach(),
			batch_txt_emb.to(device).detach(),
			batch_timestep.to(device).detach()
	)
	batch_final_reward = batch_final_reward.to(device).float()
	value_loss = F.mse_loss(
			pred_value.float().reshape([v_batch_size, 1]),
			batch_final_reward.to(device).detach().reshape([v_batch_size, 1]))
	accelerator.backward(value_loss/v_step)
	del pred_value
	del batch_state
	del batch_timestep
	del batch_final_reward
	del batch_txt_emb
	return (value_loss.item() / v_step)


@dataclasses.dataclass(frozen=False)
class TrainPolicyFuncData:
	tot_p_loss: float = 0
	tot_ratio: float = 0
	tot_kl: float = 0
	tot_grad_norm: float = 0


def _train_policy_func(
		p_batch_size,
		ratio_clip,
		reward_weight,
		kl_warmup,
		kl_weight,
		train_gradient_accumulation_steps,
		state_dict,
		pipe,
		unet_copy,
		is_ddp,
		count,
		policy_steps,
		accelerator,
		tpfdata,
		value_function
):
	"""Trains the policy function."""
	with torch.no_grad():
		indices = get_random_indices(
				state_dict["state"].shape[0], p_batch_size
		)
		batch_state = state_dict["state"][indices]
		batch_next_state = state_dict["next_state"][indices]
		batch_timestep = state_dict["timestep"][indices]
		batch_final_reward = state_dict["final_reward"][indices]
		batch_unconditional_prompt_embeds = state_dict[
				"unconditional_prompt_embeds"
		][indices]
		batch_guided_prompt_embeds = state_dict["guided_prompt_embeds"][indices]
		batch_promt_embeds = torch.cat(
				[batch_unconditional_prompt_embeds, batch_guided_prompt_embeds]
		)
		batch_txt_emb = state_dict["txt_emb"][indices]
		batch_log_prob = state_dict["log_prob"][indices]
	# calculate loss from the custom function
	# (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
	log_prob, kl_regularizer = pipe.forward_calculate_logprob(
			prompt_embeds=batch_promt_embeds.to(device),
			latents=batch_state.to(device),
			next_latents=batch_next_state.to(device),
			ts=batch_timestep.to(device),
			unet_copy=unet_copy,
			is_ddp=is_ddp,
	)
	with torch.no_grad():
		adv = batch_final_reward.to(device).reshape([p_batch_size, 1]) - value_function(
				batch_state.to(device),
				batch_txt_emb.to(device),
				batch_timestep.to(device)).reshape([p_batch_size, 1])
	ratio = torch.exp(log_prob - batch_log_prob.to(device))
	ratio = torch.clamp(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
	loss = (
			-reward_weight
			* adv.detach().float()
			* ratio.float().reshape([p_batch_size, 1])
	).mean()
	if count > kl_warmup:
		loss += kl_weight * kl_regularizer.mean()
	loss = loss / (train_gradient_accumulation_steps)
	accelerator.backward(loss)
	# logging
	tpfdata.tot_ratio += ratio.mean().item() / policy_steps
	tpfdata.tot_kl += kl_regularizer.mean().item() / policy_steps
	tpfdata.tot_p_loss += loss.item() / policy_steps