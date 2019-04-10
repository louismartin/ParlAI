from pathlib import Path
from pprint import pprint
import shlex

from fairseq import options, tasks, utils
import torch
from ts.text import word_tokenize, word_detokenize

from parlai.core.agents import Agent


def use_cuda(args):
    return torch.cuda.is_available() and not args.cpu


def simplify_sentence(sentence, model, generator, task, args):
    tokens = [task.source_dictionary.encode_line(sentence, add_if_not_exist=False).long()]
    lengths = torch.LongTensor([tokens[0].numel()])
    [batch] = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(task.max_positions(), model.max_positions()),
    ).next_epoch_itr(shuffle=False)
    if use_cuda(args):
        batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'].cuda()
        batch['net_input']['src_lengths'] = batch['net_input']['src_lengths'].cuda()

    [hypos] = task.inference_step(generator, [model], batch)
    src_tokens = utils.strip_pad(batch['net_input']['src_tokens'], task.target_dictionary.pad())
    src_str = task.source_dictionary.string(src_tokens, args.remove_bpe)
    hypo = hypos[0]
    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=hypo['tokens'].int().cpu(),
        src_str=src_str,
        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
        align_dict=None,
        tgt_dict=task.target_dictionary,
        remove_bpe=args.remove_bpe,
    )
    return hypo_str


def build_fairseq_stuff(args):
    '''Builds fairseq stuff'''
    task = tasks.setup_task(args)
    [model], _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )
    # Optimize model for generation
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda(args):
        model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    return model, generator, task, args


class TextSimplificationBotAgent(Agent):

    def __init__(self, opt, shared=None):
        exp_dir = Path('/private/home/louismartin/dev/text-simplification/experiments/fairseq/slurmjob_10080437/')
        checkpoint_path = exp_dir / 'checkpoints/checkpoint_best.pt'
        model_overrides = {'encoder_embed_path': None, 'decoder_embed_path': None}
        args_list = shlex.split(f'{exp_dir} --source-lang complex --target-lang simple --path {checkpoint_path} --beam 12 --raw-text --print-alignment --model-overrides "{model_overrides}"')  # noqa: E501
        parser = options.get_generation_parser(interactive=True)
        args = options.parse_args_and_arch(parser, args_list)
        self.model, self.generator, self.task, self.args = build_fairseq_stuff(args)
        super().__init__(opt, shared)

    def act(self):
        source_sentence = self.observation['text']
        print(source_sentence)
        source_sentence = word_tokenize(source_sentence)
        print(source_sentence)
        predicted_sentence = simplify_sentence(source_sentence, self.model, self.generator, self.task, self.args)
        print(predicted_sentence)
        predicted_sentence = word_detokenize(predicted_sentence)
        print(predicted_sentence)
        return {'text': predicted_sentence}
