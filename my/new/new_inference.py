import torch
from tqdm import tqdm

# inference test dataset
def test_prediction(model, model_name, test_loader, mask_test_origin):
    
    print('=============== Start Test Prediction ===============')

    device = 'cuda'
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outp = model(images)
            tmp_outp = outp.argmax(dim=-1)
            all_predictions.extend(tmp_outp.cpu().numpy())
    
    mask_test_origin.info_df['ans'] = all_predictions
    mask_test_origin.info_df.to_csv(f'./input/data/eval/submission_{model_name}.csv', index=False)
    print('================= Done =================')