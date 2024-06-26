
def dominates(sample_values_dominant:list,sample_values_submissive:list)->bool:
    #returns true if sample_values_dominant is dominant over sample_values_submissive
    count=0
    for metric_dom,metric_sub in zip(sample_values_dominant,sample_values_submissive):
        if metric_dom>=metric_sub:
            count+=1
    return count==len(sample_values_dominant)

def get_dominant_list(*list_args)->list:
    num_samples=len(list_args[0])
    sample_dict={}
    for n in range(num_samples):
        sample_dict[n]=[
            l[n] for l in list_args
        ]
    dominant_set=set()
    for i in range(num_samples):
        dominant=True
        for j in range(num_samples):
            if j!=i and dominates(sample_dict[j], sample_dict[i]) is True:
                dominant=False
                break
        if dominant:
            dominant_set.add(i)
    return_mega_list=[]
    for src_list in list_args:
        return_list=[src_list[d] for d in dominant_set]
        return_mega_list.append(return_list)
    return [d for d in dominant_set]