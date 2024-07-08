files=`ls merged_*`
# obtain recommenders
for file in $files ; do echo $file ; java -jar target/CBRecSys-1.0-SNAPSHOT.jar 20 $file results_generated_dataset/ ; done

## evaluate
# normal
for file in $files ; do echo $file ; java -jar target/CBRecSys-1.0-SNAPSHOT.jar 30 generated_dataset/ $file results_generated_dataset/ eval_generated_dataset/ ; done
# only users with groundtruth 
for file in $files ; do echo $file ; java -jar target/CBRecSys-1.0-SNAPSHOT.jar 31 generated_dataset/ $file results_generated_dataset/ eval_generated_dataset/ ; done

## post-eval
cd eval_generated_dataset/
for train in merged__user_item__0.2m.dat merged__user_item__0.5m.dat merged__user_item__0.8m.dat merged__user_item__no_filter.dat
do
    eval=`ls eval__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_$train
    done
done

for train in merged__user_item__0.2m.dat merged__user_item__0.5m.dat merged__user_item__0.8m.dat merged__user_item__no_filter.dat
do
    eval=`ls eval__userswith__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_userswith_$train
    done
done
cd ..

# feedback
java -jar target/CBRecSys-1.0-SNAPSHOT.jar 40 merged__user_item__0.2m.dat  results_generated_dataset/rec__merged__user_item__0.2m.dat__hkv_50_0.1_1.0_20 100 3 feedback1__merged__user_item__0.2m.dat__hkv_50_0.1_1.0_20__100_3.dat

java -jar target/CBRecSys-1.0-SNAPSHOT.jar 40 merged__user_item__0.2m.dat  results_generated_dataset/rec__merged__user_item__0.2m.dat__ub_50_1 100 3 feedback1__merged__user_item__0.2m.dat__ub_50_1__100_3.dat

for file in feedback1__merged__user_item__0.2m.dat__ub_50_1__100_3.dat ; do echo $file ; java -jar target/CBRecSys-1.0-SNAPSHOT.jar 20 $file results_feedback1_ub50/ ; done

for file in feedback1__merged__user_item__0.2m.dat__hkv_50_0.1_1.0_20__100_3.dat ; do echo $file ; java -jar target/CBRecSys-1.0-SNAPSHOT.jar 20 $file results_feedback1_hkv/ ; done

cd eval_results_feedback1_hkv/
for train in feedback1__merged__user_item__0.2m.dat__hkv_50_0.1_1.0_20__100_3.dat
do
    eval=`ls eval__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_feedback1_hkv
    done
done

for train in feedback1__merged__user_item__0.2m.dat__hkv_50_0.1_1.0_20__100_3.dat
do
    eval=`ls eval__userswith__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_userswith_feedback1_hkv
    done
done
cd ..

cd eval_results_feedback1_ub50/
for train in feedback1__merged__user_item__0.2m.dat__ub_50_1__100_3.dat
do
    eval=`ls eval__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_feedback1_ub50
    done
done

for train in feedback1__merged__user_item__0.2m.dat__ub_50_1__100_3.dat
do
    eval=`ls eval__userswith__rec__"$train"*`
    for feval in $eval
    do
        awk -v FILE=$feval '{print FILE"\t"$0}' $feval >> ../summary_userswith_feedback1_ub50
    done
done
cd ..
