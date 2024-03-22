from utils import *
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
import argparse
import nltk

nltk.download("punkt")


def train(model_name: str, model_dir: str = "./best_sum_model"):
    MODEL_NAME = model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./huggingface")

    # load dataset
    train_dataset = load_data("./data/train.csv")
    valid_dataset = load_data("./data/valid.csv")

    y_train = train_dataset["summary"].values
    y_valid = valid_dataset["summary"].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer)

    # make dataset for pytorch
    SUM_train_dataset = SUM_Dataset(tokenized_train, y_train)
    SUM_valid_dataset = SUM_Dataset(tokenized_valid, y_valid)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    # setting model hyperparameter
    sum_model_config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir="./huggingface")

    sum_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, config=sum_model_config, cache_dir="./huggingface"
    )
    sum_model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        seed=42,
        save_total_limit=1,
        save_strategy="epoch",
        # save_steps=500,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1,
        num_train_epochs=10,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro f1 score",  # 수정 필요
        greater_is_better=True,  # if loss False
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
    )

    # trainer
    trainer = Trainer(
        model=sum_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=SUM_train_dataset,  # training dataset
        eval_dataset=SUM_valid_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[early_stopping],
    )

    # train model
    trainer.train()
    sum_model.save(model_dir)


def inference(tokenizer_name: str, model_dir: str, text: str, fine_tuned: bool = False):

    Tokenizer_NAME = tokenizer_name
    MODEL_NAME = model_dir

    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if fine_tuned:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(Tokenizer_NAME)

    max_input_length = 512

    inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, return_tensors="pt"
    )
    output = model.generate(
        **inputs, num_beams=8, do_sample=True, min_length=10, max_length=100
    )
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    print(predicted_title)


if __name__ == "__main__":

    parser = argparse.AugumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--fine_tuned", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_name", type=str, default="lcw99/t5-base-korean-text-summary"
    )
    parser.add_argument("--model_dir", type=str, default="./best_sum_model/")
    args = parser.parse_args()

    set_seed(args.seed)

    # train
    if args.train:
        train(model_name=args.model_name, model_dir=args.model_dir)
    else:  # inference
        # text = input()
        text = """
        주인공 강인구(하정우)는 ‘수리남에서 홍어가 많이 나는데 다 갖다버린다’는 친구 
        박응수(현봉식)의 얘기를 듣고 수리남산 홍어를 한국에 수출하기 위해 수리남으로 간다. 
        국립수산과학원 측은 “실제로 남대서양에 홍어가 많이 살고 아르헨티나를 비롯한 남미 국가에서 홍어가 많이 잡힌다”며 
        “수리남 연안에도 홍어가 많이 서식할 것”이라고 설명했다.

        그러나 관세청에 따르면 한국에 수리남산 홍어가 수입된 적은 없다. 
        일각에선 “돈을 벌기 위해 수리남산 홍어를 구하러 간 설정은 개연성이 떨어진다”는 지적도 한다. 
        드라마 배경이 된 2008~2010년에는 이미 국내에 아르헨티나, 칠레, 미국 등 아메리카산 홍어가 수입되고 있었기 때문이다. 
        실제 조봉행 체포 작전에 협조했던 ‘협력자 K씨’도 홍어 사업이 아니라 수리남에 선박용 특수용접봉을 파는 사업을 하러 수리남에 갔었다.
        """

        inference(
            tokenizer_name=args.model_name,
            model_dir=args.model_dir,
            text=text,
            fine_tuned=args.fine_tuned,
        )
