from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "/data/models/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = """
            Hi, my name is Mari Takahashi. I play video games for a living.
            I share a birthday with the Cookie Monster. I love sweets. It's my sweet spot.
            And that's why I'm qualified to judge these boys on their pies.
            Hi, my name is Jessica Schupak. I'm a marketing consultant for restaurants.
            I basically kind of judge with restaurants if something should go on the menu or not.
            I'm Brianna Abrams. I am the chief pie Pie Smith at Winston Pies here in Los Angeles.
            We're really excited to have you here
            and talk about holiday pies,
            because I love making pies and I love talking about pies.
            It is unbelievably hard to make a pie,
            especially if you're just doing it one time on a Saturday.
            You spend hours making the dough and letting it rest,
            hours making the filling, then you're watching it bake,
            it comes out and maybe it doesn't look making the filling, then you're watching it bake, it comes out,
            maybe it doesn't look so pretty,
            but then it tastes amazing or it looks beautiful
            and it is not delicious.
            Perfecting pie is a challenge.
            Pies.
            My baking experience is limited to eating raw cookie dough
            and the last time we made one of these videos.
            Last time on Without a Recipe,
            my bread looked amazing and tasted very plain.
            So this time I'm not gonna forget the flavor.
            I like to pride myself with the fact
            that I'm good at making food for others.
            I engineered this video series
            to be a competition show that I could win,
            and then I didn't win.
            Last time I ended up winning.
            I'm just gonna relax, see what comes to my brain,
            and hopefully we have a similar outcome.
          """
inputs = tokenizer(article, return_tensors="pt")

translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zho_Hans"], max_length=512
)
outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
print(outputs)


