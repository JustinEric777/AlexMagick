from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_sequence = """
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

pipeline_ins = pipeline(task=Tasks.translation, model="/data/models/damo-csanmt-en-zh-large/")
outputs = pipeline_ins(input=input_sequence)

print(outputs['translation'])
